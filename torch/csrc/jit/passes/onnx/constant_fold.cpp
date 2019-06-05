#include <torch/csrc/jit/passes/onnx/constant_fold.h>
#include <c10/util/Exception.h>

#include <c10/util/Optional.h>
#include <algorithm>

namespace torch {
namespace jit {

namespace onnx {
using namespace ::c10::onnx;
}

namespace {

using ParamMap = std::map<std::string, at::Tensor>;
using ValueToParamPairMap =
    std::map<Value*, std::pair<std::string, at::Tensor>>;

std::unordered_map<int, at::ScalarType> onnxTypeToScalarTypeMap = {
    // Only conversion of ONNX numeric types is included here.
    // Unsigned ONNX types are mapped to the next higher signed
    // ScalarType type.
    {1, at::kFloat},
    {2, at::kByte},
    {3, at::kChar},
    {4, at::kInt},
    {5, at::kShort},
    {6, at::kInt},
    {7, at::kLong},
    {10, at::kFloat},
    {11, at::kDouble},
    {12, at::kLong},
};

ValueToParamPairMap buildValueToParamsMap(
    Block* b,
    const ParamMap& paramsDict) {
  ValueToParamPairMap valsToParamsMap;
  for (auto& input : b->inputs()) {
    auto it = paramsDict.find(input->uniqueName());
    if (it != paramsDict.end()) {
      valsToParamsMap.emplace(input, *it);
    }
  }
  return valsToParamsMap;
}

void buildParamsMapFromValueToParamsMap(
    const ValueToParamPairMap& valsToParamsMap,
    ParamMap& paramsDict) {
  paramsDict.clear();
  for (const auto& nameTensorParamPair : valsToParamsMap) {
    paramsDict.insert(nameTensorParamPair.second);
  }
}

void eraseUnusedBlockInputs(Block* b) {
  for (size_t i_1 = b->inputs().size(); i_1 > 0; --i_1) {
    size_t i = i_1 - 1;
    if (!b->inputs().at(i)->hasUses()) {
      b->eraseInput(i);
    }
  }
}

c10::optional<at::Tensor> runTorchBackendForOnnx(
    const Node* node,
    std::vector<at::Tensor>& inputTensorValues) {
  at::Tensor updated_val;
  if (node->kind() == onnx::Slice) {
    assert(inputTensorValues.size() == 1);
    if (!(node->hasAttributeS("starts") && node->hasAttributeS("ends"))) {
      return c10::nullopt;
    }
    auto startsAttr = node->is(attr::starts);
    auto endsAttr = node->is(attr::ends);
    if (startsAttr.size() != endsAttr.size()) {
      return c10::nullopt;
    }
    std::vector<int64_t> axesAttr;
    if (node->hasAttributeS("axes")) {
      axesAttr = node->is(attr::axes);
    } else {
      axesAttr.resize(startsAttr.size());
      std::iota(axesAttr.begin(), axesAttr.end(), 0);
    }
    updated_val = inputTensorValues[0];
    for (size_t i = 0; i < axesAttr.size(); ++i) {
      updated_val = at::narrow(
          updated_val, axesAttr[i], startsAttr[i], endsAttr[i] - startsAttr[i]);
    }
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Concat) {
    if (!node->hasAttributeS("axis")) {
      return c10::nullopt;
    }
    updated_val =
        at::cat(at::TensorList(inputTensorValues), node->i(attr::axis));
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Unsqueeze) {
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("axes")) {
      return c10::nullopt;
    }
    updated_val = inputTensorValues[0];
    for (auto axis : node->is(attr::axes)) {
      updated_val = at::unsqueeze(updated_val, axis);
    }
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Transpose) {
    assert(inputTensorValues.size() == 1);
    if (!node->hasAttributeS("perm")) {
      return c10::nullopt;
    }
    updated_val = inputTensorValues[0].permute(node->is(attr::perm));
    return c10::optional<at::Tensor>(updated_val);
  } else if (node->kind() == onnx::Cast) {
    assert(inputTensorValues.size() == 1);
    if (node->hasAttributeS("to") &&
        onnxTypeToScalarTypeMap.find(node->i(attr::to)) !=
            onnxTypeToScalarTypeMap.end()) {
      updated_val =
          inputTensorValues[0].to(onnxTypeToScalarTypeMap[node->i(attr::to)]);
      return c10::optional<at::Tensor>(updated_val);
    }
    return c10::nullopt;
  } else {
    return c10::nullopt;
  }
}

bool isConstant(Value* val, const ValueToParamPairMap& valsToParamsMap) {
  auto parentNode = val->node();
  return (parentNode->kind() == prim::Param &&
          valsToParamsMap.find(val) !=
              valsToParamsMap
                  .end()) || // Checks val is a parameter and not a real input
      (parentNode->kind() == onnx::Constant && !parentNode->mustBeNone() &&
       parentNode->kindOf(attr::value) ==
           AttributeKind::t); // Check other types?
}

std::vector<at::Tensor> getValues(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  size_t numInputs = node->inputs().size();
  std::vector<at::Tensor> inputTensorValues;
  inputTensorValues.reserve(numInputs);
  for (auto val : node->inputs()) {
    if (val->node()->kind() == prim::Param) {
      auto itr = valsToParamsMap.find(val);
      if (itr == valsToParamsMap.end()) {
        throw std::runtime_error(
            "getValues: Input value not found amongst constant parameters.");
      }
      inputTensorValues.push_back(itr->second.second);
    } else if (val->node()->kind() == onnx::Constant) {
      inputTensorValues.push_back(val->node()->t(attr::value));
    } else {
      throw std::runtime_error(
          "getValues: Unsupported kind of constant node found.");
    }
  }
  AT_ASSERT(inputTensorValues.size() == numInputs);
  return inputTensorValues;
}

void eraseUnusedValuesFromMap(ValueToParamPairMap& valsToParamsMap) {
  auto it = valsToParamsMap.begin();
  while (it != valsToParamsMap.end()) {
    if (!it->first->hasUses()) {
      it = valsToParamsMap.erase(it);
    } else {
      ++it;
    }
  }
}

bool areNodeInputsConstant(
    Node* node,
    const ValueToParamPairMap& valsToParamsMap) {
  return std::all_of(
      node->inputs().begin(),
      node->inputs().end(),
      [&valsToParamsMap](Value* v) { return isConstant(v, valsToParamsMap); });
}

std::vector<Node*> getOnnxConstParentsToRemove(Node* node) {
  std::vector<Node*> parentNodes;
  for (auto val : node->inputs()) {
    // If the parent of 'node' is an onnx::Constant node,
    // and 'node' is the only downstream node it serves (this
    // is important), then push it in the list to remove.
    if (val->node()->kind() == onnx::Constant &&
        val->uses().size() == 1) {
          parentNodes.push_back(val->node());
    }
  }
  return parentNodes;
}

} // Anonymous namespace


// Recursive collector of a tree having Concat as a root and Constants as leaves.
// With Unsqueeze, Gather and Shape only in between.
// Returns 1 if succeeded.
int collectFoldables(int level, Node* node, std::vector<std::vector<Node*>>& removeNodes,
    std::vector<at::Tensor>& inputTensorValues) {
  if (level > 3) {
    return -1; // no deeper
  }

  std::cerr << "kind: " << node->kind().toDisplayString() << "  ";
  node->print(std::cerr, level, nullptr);
  if (node->kind() == prim::Param) {
    return -2;
  }

  int ret = 0;
  if (node->kind() == onnx::Concat ||
      node->kind() == onnx::Unsqueeze ||
      node->kind() == onnx::Gather ||
      node->kind() == onnx::Shape ||
      node->kind() == onnx::Constant) {
    if (removeNodes.size() <= level) {
      removeNodes.emplace_back(std::vector<Node*>());
    }
    removeNodes[level].emplace_back(node);

    if (node->kind() == onnx::Constant) {
      auto names = node->attributeNames();
      for (auto name : names) {
        if (name == attr::value && node->kindOf(name) == AttributeKind::t) {
          at::Tensor val = node->t(name).dim() == 0 ? node->t(name).unsqueeze(0) : node->t(name);
          std::cout << std::string(level, ' ') << val.toString() << " "
              << val.sizes() << " : " << val.item().toLong() << std::endl;
          inputTensorValues.push_back(val);
          ret = 1;
          break;
        }
      }
    } else {
      for (auto inp : node->inputs()) {
        ret = collectFoldables(level + 1, inp->node(), removeNodes, inputTensorValues);
        if (ret == 0 || ret == -2) {
          break;
        }
      }
    }
  } else {
    ret = -1;
  }
  return ret;
}

// This method updates the block in-place to fold all the one-time
// constant-based computations/ops into an initializer node.
void ConstantFoldONNX(Block* b, ParamMap& paramsDict) {
  AT_ASSERT(b->param_node());
/*
 * We can do better for the cases like this:
 *
  %30 : Float(2, 256, 6, 6) = onnx::AveragePool[kernel_shape=[1, 1], strides=[1, 1]](%29), scope: AlexNet/AdaptiveAvgPool2d[avgpool]
  %31 : Long() = onnx::Constant[value={2}](), scope: AlexNet
  %32 : Long() = onnx::Constant[value={9216}](), scope: AlexNet
  %33 : Tensor = onnx::Unsqueeze[axes=[0]](%31)
  %34 : Tensor = onnx::Unsqueeze[axes=[0]](%32)
  %35 : Tensor = onnx::Concat[axis=0](%33, %34)
  %36 : Float(2, 9216) = onnx::Reshape(%30, %35), scope: AlexNet/Sequential[classifier]/Dropout[0]

  or this:

  %40 : Tensor = onnx::Shape(%27)
  %41 : Tensor = onnx::Constant[value={1}]()
  %42 : Tensor = onnx::Gather(%40, %41)
  %43 : Tensor = onnx::Unsqueeze[axes=[0]](%42)
  %44 : Tensor = onnx::Constant[value={3}]()
  %45 : Tensor = onnx::Constant[value={2}]()
  %46 : Tensor = onnx::Unsqueeze[axes=[0]](%45)
  %47 : Tensor = onnx::Concat[axis=0](%46, %43, %44)
  %48 : Tensor = onnx::ConstantOfShape(%47)


  %25 : Long() = onnx::Constant[value={1}](), scope: RNN
  %26 : Tensor = onnx::Shape(%input), scope: RNN
  %27 : Long() = onnx::Gather[axis=0](%26, %25), scope: RNN
  %28 : Long() = onnx::Constant[value={6}](), scope: RNN
  %29 : Long() = onnx::Constant[value={3}](), scope: RNN
  %30 : Tensor = onnx::Unsqueeze[axes=[0]](%28)
  %31 : Tensor = onnx::Unsqueeze[axes=[0]](%27)
  %32 : Tensor = onnx::Unsqueeze[axes=[0]](%29)
  %33 : Tensor = onnx::Concat[axis=0](%30, %31, %32)
  %34 : Float(6, 7, 3) = onnx::ConstantOfShape[value={0}](%33), scope: RNN

  %15 : Tensor = onnx::Transpose[perm=[1, 0, 2]](%input.1), scope: RnnModelWithPackedSequence
  %21 : Tensor = onnx::Shape(%15)
  %22 : Tensor = onnx::Constant[value={1}]()
  %23 : Tensor = onnx::Gather(%21, %22)
  %24 : Tensor = onnx::Unsqueeze[axes=[0]](%23)
  %25 : Tensor = onnx::Constant[value={3}]()
  %26 : Tensor = onnx::Constant[value={1}]()
  %27 : Tensor = onnx::Unsqueeze[axes=[0]](%26)
  %28 : Tensor = onnx::Concat[axis=0](%27, %24, %25)
  %29 : Tensor = onnx::ConstantOfShape(%28)

*/

//onnx::Slice[axes=[0], ends=[1], starts=[0]](%30), scope: AlexNet

  int collected = 0;
  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
    c10::optional<at::Tensor> updatedValWrapped;
    std::vector<at::Tensor> inputTensorValues;
    std::vector<std::vector<Node*>> removeNodes;
    auto node = *it;
    if (node->kind() == onnx::Concat && node->hasUses()) {
      collected = collectFoldables(0, node, removeNodes, inputTensorValues);
      if (collected == 1) {
        updatedValWrapped = runTorchBackendForOnnx(node, inputTensorValues);
        if (updatedValWrapped == c10::nullopt) {
          // Constant folding is not supported for this op. Skip it.
          collected = 0;
          continue;
        }
        at::Tensor updatedVal = *updatedValWrapped;
        Node* new_shape = b->owningGraph()->create(onnx::Constant, 1);
        new_shape->t_(attr::value, updatedVal);
        auto newSourceNodeOutput = new_shape->insertAfter(node)->output();
        newSourceNodeOutput->inferTypeFrom(updatedVal);
        node->outputs().at(0)->replaceAllUsesWith(newSourceNodeOutput);
        node->removeAllInputs();
        for (auto itn = removeNodes.begin(); itn != removeNodes.end(); ++itn) {
          for (auto n : *itn) {
            if(node != n) {
              n->destroy();
            }
          }
        }
        it.destroyCurrent();
      }
    }
  }
  if (collected == 1) {
    // std::cout << b->owningGraph()->toString() << std::endl;
    return;
  }

  // Default implementation
//  auto valsToParamsMap = buildValueToParamsMap(b, paramsDict);
//  // Only the root block is constant-folded. Folding nested blocks is
//  // not supported for now.
//  for (auto it = b->nodes().begin(), end = b->nodes().end(); it != end; ++it) {
//    auto node = *it;
//    if (node->outputs().size() > 1) {
//      // Constant folding for multiple-output nodes not supported. Skip it.
//      continue;
//    }
//    if (!areNodeInputsConstant(node, valsToParamsMap)) {
//      // If all the inputs to this node are not either parameter or
//      // onnx::Constant, then skip this node.
//      continue;
//    }
//    auto inputTensorValues = getValues(node, valsToParamsMap);
//    if (inputTensorValues.empty()) {
//      // This is a terminal node with no inputs, such as onnx::Constant. Skip
//      // it.
//      continue;
//    }
//    auto updatedValWrapped = runTorchBackendForOnnx(node, inputTensorValues);
//    if (updatedValWrapped == c10::nullopt) {
//      // Constant folding is not supported for this op. Skip it.
//      continue;
//    }
//    // Create a new input to the block (prim::Param node output). Add a
//    // corresponding entryin valToParamMap. Replace the downstream inputs
//    // with this value, and disconnect all the input values of the folded node.
//    at::Tensor updatedVal = *updatedValWrapped;
//    auto newSourceNodeOutput = b->addInput();
//    valsToParamsMap.insert(
//        {newSourceNodeOutput,
//         std::make_pair(newSourceNodeOutput->uniqueName(), updatedVal)});
//    newSourceNodeOutput->inferTypeFrom(updatedVal);
//    node->outputs().at(0)->replaceAllUsesWith(newSourceNodeOutput);
//
//    // Next we remove the current node that has been replaced by
//    // an initializer. But before we start de-wiring this node,
//    // we check if any parents of this nodes were onnx::Constant
//    // and remove them first (following proper sequence as shown
//    // below), and then remove the current node. If the parent was
//    // an initializer (not onnx::Constant) then they are all removed
//    // by eraseUnusedBlockInputs() call (below) outside the loop.
//    auto onnxConstParents = getOnnxConstParentsToRemove(node);
//    node->removeAllInputs();
//    for (auto* n : onnxConstParents) {
//      n->destroy();
//    }
//    it.destroyCurrent();
//  }
//  eraseUnusedValuesFromMap(valsToParamsMap);
//  eraseUnusedBlockInputs(b);
//  buildParamsMapFromValueToParamsMap(valsToParamsMap, paramsDict);
}

} // namespace jit
} // namespace torch
