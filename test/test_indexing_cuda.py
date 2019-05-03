import torch
import time
import copy
import numpy as np
from common_utils import TestCase, run_tests
from test_indexing import TestIndexing, NumpyTests

class TestIndexingCuda(TestIndexing):
    def setUp(self):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    def test_advanced_indexing1d(self):
        for d1 in torch.arange(21, 32):
            x = np.random.random((d1)).astype(np.float32)
            xcpu = torch.tensor(copy.deepcopy(x), requires_grad=True, device='cpu')
            xcuda = torch.tensor(copy.deepcopy(x), requires_grad=True, device='cuda')

            ycpu = xcpu[[7,9,19,20]]
            ycuda = xcuda[[7,9,19,20]]
            ycpu.backward(gradient=torch.ones_like(ycpu))
            ycuda.backward(gradient=torch.ones_like(ycuda))
            self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

            ycpu = xcpu[[0,1]]
            ycuda = xcuda[[0,1]]
            ycpu.backward(gradient=torch.ones_like(ycpu))
            ycuda.backward(gradient=torch.ones_like(ycuda))
            self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

            ycpu = xcpu[[0]]
            ycuda = xcuda[[0]]
            ycpu.backward(gradient=torch.ones_like(ycpu))
            ycuda.backward(gradient=torch.ones_like(ycuda))
            self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

            ycpu = xcpu[[0,1,0,7]]
            ycuda = xcuda[[0,1,0,7]]
            ycpu.backward(gradient=torch.ones_like(ycpu))
            ycuda.backward(gradient=torch.ones_like(ycuda))
            self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

            ycpu = xcpu[:]
            ycuda = xcuda[:]
            ycpu.backward(gradient=torch.ones_like(ycpu))
            ycuda.backward(gradient=torch.ones_like(ycuda))
            self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

    def test_advanced_indexing2d(self):
        for d1 in torch.arange(21, 32):
            for d2 in torch.arange(10, 15):
                x = np.random.random((d1,d2)).astype(np.float32)
                xcpu = torch.tensor(copy.deepcopy(x), requires_grad=True, device='cpu')
                xcuda = torch.tensor(copy.deepcopy(x), requires_grad=True, device='cuda')

                ycpu = xcpu[:,[7,9]]
                ycuda = xcuda[:,[7,9]]
                ycpu.backward(gradient=torch.ones_like(ycpu))
                ycuda.backward(gradient=torch.ones_like(ycuda))
                self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                ycpu = xcpu[[0,1]]
                ycuda = xcuda[[0,1]]
                ycpu.backward(gradient=torch.ones_like(ycpu))
                ycuda.backward(gradient=torch.ones_like(ycuda))
                self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                ycpu = xcpu[:,[0]]
                ycuda = xcuda[:,[0]]
                ycpu.backward(gradient=torch.ones_like(ycpu))
                ycuda.backward(gradient=torch.ones_like(ycuda))
                self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                ycpu = xcpu[[0,1,0,7]]
                ycuda = xcuda[[0,1,0,7]]
                ycpu.backward(gradient=torch.ones_like(ycpu))
                ycuda.backward(gradient=torch.ones_like(ycuda))
                self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                ycpu = xcpu[[1,0,1],:]
                ycuda = xcuda[[1,0,1],:]
                ycpu.backward(gradient=torch.ones_like(ycpu))
                ycuda.backward(gradient=torch.ones_like(ycuda))
                self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

    def test_advanced_indexing3d(self):
        for d1 in torch.arange(21, 32):
            for d2 in torch.arange(10, 15):
                for d3 in torch.arange(8, 10):
                    x = np.random.random((d1,d2,d3)).astype(np.float32)
                    xcpu = torch.tensor(copy.deepcopy(x), requires_grad=True, device='cpu')
                    xcuda = torch.tensor(copy.deepcopy(x), requires_grad=True, device='cuda')

                    ycpu = xcpu[:,[7,9]]
                    ycuda = xcuda[:,[7,9]]
                    ycpu.backward(gradient=torch.ones_like(ycpu))
                    ycuda.backward(gradient=torch.ones_like(ycuda))
                    self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                    ycpu = xcpu[[0,1],:,[7,4]]
                    ycuda = xcuda[[0,1],:,[7,4]]
                    ycpu.backward(gradient=torch.ones_like(ycpu))
                    ycuda.backward(gradient=torch.ones_like(ycuda))
                    self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                    ycpu = xcpu[:,[0,1,7,1],:]
                    ycuda = xcuda[:,[0,1,7,1],:]
                    ycpu.backward(gradient=torch.ones_like(ycpu))
                    ycuda.backward(gradient=torch.ones_like(ycuda))
                    self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                    ycpu = xcpu[:,[0,1,0,7]]
                    ycuda = xcuda[:,[0,1,0,7]]
                    ycpu.backward(gradient=torch.ones_like(ycpu))
                    ycuda.backward(gradient=torch.ones_like(ycuda))
                    self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                    ycpu = xcpu[[1,0,1],[1,0,1],:]
                    ycuda = xcuda[[1,0,1],[1,0,1],:]
                    ycpu.backward(gradient=torch.ones_like(ycpu))
                    ycuda.backward(gradient=torch.ones_like(ycuda))
                    self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                    ycpu = xcpu[[1,1,1],[1,1,1],:]
                    ycuda = xcuda[[1,1,1],[1,1,1],:]
                    ycpu.backward(gradient=torch.ones_like(ycpu))
                    ycuda.backward(gradient=torch.ones_like(ycuda))
                    self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

    def test_advanced_indexing4d(self):
        for d1 in torch.arange(2, 5):
            for d2 in torch.arange(2, 5):
                for d3 in torch.arange(2, 5):
                    for d4 in torch.arange(25, 63):
                        x = np.random.random((d1,d2,d3,d4)).astype(np.float32)
                        xcpu = torch.tensor(copy.deepcopy(x), requires_grad=True, device='cpu')
                        xcuda = torch.tensor(copy.deepcopy(x), requires_grad=True, device='cuda')

                        ycpu = xcpu[[0,1],:,[0],[7,11]]
                        ycuda = xcuda[[0,1],:,[0],[7,11]]
                        ycpu.backward(gradient=torch.ones_like(ycpu))
                        ycuda.backward(gradient=torch.ones_like(ycuda))
                        self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                        ycpu = xcpu[:,[0,1,1,1],:]
                        ycuda = xcuda[:,[0,1,1,1],:]
                        ycpu.backward(gradient=torch.ones_like(ycpu))
                        ycuda.backward(gradient=torch.ones_like(ycuda))
                        self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                        ycpu = xcpu[:,[0,1,0,1]]
                        ycuda = xcuda[:,[0,1,0,1]]
                        ycpu.backward(gradient=torch.ones_like(ycpu))
                        ycuda.backward(gradient=torch.ones_like(ycuda))
                        self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                        ycpu = xcpu[[1,0,1],[1,0,1],:]
                        ycuda = xcuda[[1,0,1],[1,0,1],:]
                        ycpu.backward(gradient=torch.ones_like(ycpu))
                        ycuda.backward(gradient=torch.ones_like(ycuda))
                        self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

                        ycpu = xcpu[[1,1,1],[1,1,1],:]
                        ycuda = xcuda[[1,1,1],[1,1,1],:]
                        ycpu.backward(gradient=torch.ones_like(ycpu))
                        ycuda.backward(gradient=torch.ones_like(ycuda))
                        self.assertEqual(0, (xcpu.grad - xcuda.grad.cpu()).abs().sum())

    def test_advanced_indexing_benchmark(self):
        Ns=[16, 64, 128, 256, 512]
        Ds=[1, 16, 256, 1024, 4096]
        Ks=[1, 16, 256, 1024, 4096]
        iters=[100, 50, 20, 10, 5]
        f_better = []
        g_better = []
        sames, f_speedups, b_speedups = [], [], []
        rows = []
        deltas = []
        y_diffs = []
        i = 0

        for N in Ns:
            for D in Ds:
                for K in Ks:
                    y_diff, f_speedup, f_time_us, g_time_us =\
                        self.benchmark(N, D, K, y_diffs, iters[i])
                    if f_speedup < 1.0:
                        f_better.append((N, D, K, f_time_us, g_time_us))
                    else:
                        g_better.append((N, D, K, f_time_us, g_time_us))
                    rows.append((N, D, K, f_time_us, g_time_us))
                    deltas.append(f_time_us - g_time_us)
                    f_speedups.append(f_speedup)
            i = i + 1
        # These algos should have similar complexities
        self.assertGreater(np.min(f_speedups), 0.6)
        self.assertLess(np.max(f_speedups), 3)
        #...and deliver identical results
        self.assertEqual(0, np.abs(y_diff).sum())

    def benchmark(self, N, D, K, y_diffs, iters):
        index_times, gather_times = [], []
        for _ in range(1):
            x = np.random.random((N,D)).astype(np.float32)
            xi = torch.tensor(copy.deepcopy(x), requires_grad=True, device='cuda')
            xg = torch.tensor(copy.deepcopy(x), requires_grad=True, device='cuda')
            ind = np.random.randint(0, N, size=(K,))
            indi = torch.tensor(copy.deepcopy(ind), device='cuda')
            indg = torch.tensor(copy.deepcopy(ind), device='cuda')


            # Time forward / backward for index
            y_index, t_index = self.timeit(self.index, xi, indi, iters)
            y_gather, t_gather = self.timeit(self.gather, xg, indg, iters)

            index_times.append(t_index)
            gather_times.append(t_gather)

            with torch.no_grad():
                y_diff = (xi.grad - xg.grad).abs().sum()
                y_diffs.append(y_diff.item())

        y_diff = np.max(y_diffs)
        t_index = np.mean(index_times)
        t_gather = np.mean(gather_times)

        speedup = t_index / t_gather

        return y_diff, speedup, t_index, t_gather

    def index(self, x, idx):
        return x[idx]
    #  return torch.nn.functional.embedding(idx,x)

    def gather(self, x, idx):
        idx = idx[:, None].expand(idx.shape[0], x.shape[1])
        return x.gather(0, idx)

    def timeit(self, f, x, idx, iters):
        if x.grad is not None:
            x.grad.data.zero_()

        t0 = time.time()
        y = f(x, idx)
        dy = torch.ones_like(y)
        y.backward(gradient=dy)
        if x.is_cuda:
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(iters):
            y = f(x, idx)
            y.backward(gradient=dy)
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.time()

        # in microseconds
        t_us = 1000000.0 * (t1 - t0) / iters
        return y, t_us


class NumpyTestsCuda(NumpyTests):
    def setUp(self):
        torch.set_default_tensor_type(torch.cuda.FloatTensor)


if __name__ == '__main__':
    if torch.cuda.is_available():
        run_tests()
    else:
        print("Skipping test_indexing_cuda.py")
