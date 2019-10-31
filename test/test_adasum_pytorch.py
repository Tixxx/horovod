# Copyright 2019 Microsoft. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import torch
import horovod.torch as hvd
import numpy as np
import time
from horovod.torch.mpi_ops import synchronize
import os
import math
import unittest
import warnings
from distutils.version import LooseVersion

_fp16_supported = LooseVersion(torch.__version__) >= LooseVersion('1.0.0')

class TorchAdasumTests(unittest.TestCase):
  """
  Tests for Adasum reduction logic in horovod.torch.
  """
  def __init__(self, *args, **kwargs):
    super(TorchAdasumTests, self).__init__(*args, **kwargs)
    warnings.simplefilter('module')
    hvd.init()
    self.device = torch.device('cuda:{}'.format(hvd.local_rank()))
    np.random.seed(2)
    torch.manual_seed(2)
    self.size = hvd.size()
    self.local_size = hvd.local_size()
    self.rank = hvd.rank()

    self.data_types = [np.float32]
    if _fp16_supported:
      self.data_types.append(np.float16)

  def diff_ratio(self, true_vec, comp_vec):
    norm_diff = np.linalg.norm(true_vec-comp_vec)
    norm_true = np.linalg.norm(true_vec)
    return norm_diff/norm_true/100.

  def are_close(self, data_type, true_vec, comp_vec):
    return self.diff_ratio(true_vec, comp_vec) < np.finfo(data_type).eps

  def test_orthogonal(self):
    for data_type in self.data_types:
      denominator = self.local_size
      all_Ns = [self.size*20 - 17, self.size*2+1, self.size+2, 2**19]
      tensors = []
      all_qs = []
      for N in all_Ns:
        a = np.random.normal(0, 1, (N,self.size)).astype(np.float64)
        q, r = np.linalg.qr(a)
        q = q.astype(data_type)
        all_qs.append(q.astype(np.float64))
        tensors.append(q[:,hvd.rank()])

      tensors = list(map(lambda x: torch.from_numpy(x).to(self.device), tensors))

      handles = [
        hvd.allreduce_async(tensor, op=hvd.Adasum)
        for tensor in tensors
      ]

      reduced_tensors = [synchronize(h) for h in handles]

      expected = [np.sum(q,axis=1) / denominator for q in all_qs]
      all_comp = [self.are_close(data_type, e, rt.cpu().numpy()) for e,rt in zip(expected,reduced_tensors)]
      if np.alltrue(all_comp):
        print('Orthogonal test passed')
      else:
        for c,e,rt in zip(all_comp, expected, reduced_tensors):
          if c == False:
            print('computed: ', rt)
            print('expected: ', e)
            print('off by: ', self.diff_ratio(e,rt.cpu().numpy()))
            self.assertFalse()          

  def test_parallel(self):
    for data_type in self.data_types:
      all_Ns = [self.size*20 - 13, self.size*2+1, self.size+2, 2**19]
      tensors = []
      all_qs = []
      for N in all_Ns:
        a = np.random.normal(0, 1, (N, 1)).astype(np.float64)
        r = np.random.normal(0, 1, (self.size, 1)).astype(np.float64)
        q = np.dot(a,r.T)
        q = q.astype(data_type)
        all_qs.append(q.astype(np.float64))
        tensors.append(q[:,hvd.rank()])

      tensors = list(map(lambda x: torch.from_numpy(x).to(self.device), tensors))

      handles = [
        hvd.allreduce_async(tensor, op=hvd.Adasum)
        for tensor in tensors
      ]

      reduced_tensors = [synchronize(h) for h in handles]

      expected = [np.sum(q,axis=1) / self.size for q in all_qs]
      all_comp = [self.are_close(data_type, e, rt.cpu().numpy()) for e,rt in zip(expected,reduced_tensors)]
      if np.alltrue(all_comp):
        print('Parallel test passed')
      else:
        for c,e,rt in zip(all_comp, expected, reduced_tensors):
          if c == False:
            print('computed: ', rt)
            print('expected: ', e)
            print('off by: ', self.diff_ratio(e,rt.cpu().numpy()))
            self.assertFalse()          

  def test_stability(self):
    for data_type in self.data_types:
      N = 1024
      a = np.random.normal(0, np.finfo(data_type).tiny, (N, 1)).astype(np.float64)
      r = np.random.normal(0, 1, (self.size, 1)).astype(np.float64)
      q = np.dot(a,r.T).astype(data_type).astype(np.float64)
      tensor = np.zeros(N,dtype=data_type)
      tensor[:] = q[:,hvd.rank()]

      tensor = torch.from_numpy(tensor).to(self.device)

      hvd.allreduce_(tensor, op=hvd.Adasum)

      expected = np.sum(q,axis=1) / self.size
      comp = self.are_close(data_type, expected, tensor.cpu().numpy()) 
      if comp:
        print('Stability test passed')
      else:
        print('computed: ', tensor)
        print('expected: ', expected)
        print('off by: ', self.diff_ratio(expected,tensor.cpu().numpy()))
        self.assertFalse()          

  def test_stability_2(self):
    for data_type in self.data_types:
      N = 1024
      dt_min = np.finfo(data_type).tiny.astype(np.float64)
      dt_max = math.sqrt(np.finfo(data_type).max.astype(np.float64))
      a = np.random.normal(0, 1, (N, 1)).astype(np.float64)
      r = np.array([dt_max**(float(i+1)/float(self.size))*dt_min**(float(self.size-i-1)/float(self.size)) for i in range(self.size)]).reshape(self.size,1).astype(np.float64)
      np.random.shuffle(r)
      q = np.dot(a,r.T).astype(data_type).astype(np.float64)
      tensor = np.zeros(N,dtype=data_type)
      tensor[:] = q[:,hvd.rank()]

      tensor = torch.from_numpy(tensor).to(self.device)

      hvd.allreduce_(tensor, op=hvd.Adasum)

      expected = np.sum(q,axis=1) / self.size
      comp = self.are_close(data_type, expected, tensor.cpu().numpy()) 
      if comp:
        print('Stability 2 test passed')
      else:
        print('computed: ', tensor)
        print('expected: ', expected)
        print('off by: ', self.diff_ratio(expected,tensor.cpu().numpy()))
        self.assertFalse()          

if __name__ == "__main__":
   unittest.main()