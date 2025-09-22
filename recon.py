from __future__ import annotations
from dataclasses import dataclass
import numpy as np

SUB = 0
SUR = 1
POR = 2
RET = 3
GEN = 4

@dataclass
class ReCoN:
    num: int = 0

    # adjacency matrix of weights, shape: (num, num), float
    w_sub: np.ndarray = None
    w_sur: np.ndarray = None
    w_por: np.ndarray = None
    w_ret: np.ndarray = None
    w_gen: np.ndarray = None

    # por/ret existence checks: shape (num, ), bool
    has_por: np.ndarray = None 
    has_ret: np.ndarray = None 

    # activations shape: (num, ) float
    a_sub: np.ndarray = None
    a_sur: np.ndarray = None
    a_por: np.ndarray = None
    a_ret: np.ndarray = None
    a_gen: np.ndarray = None

    def __post_init__(self):
        self.w_sub = np.zeros((self.num, self.num), dtype=float)
        self.w_sur = np.zeros((self.num, self.num), dtype=float)
        self.w_por = np.zeros((self.num, self.num), dtype=float)
        self.w_ret = np.zeros((self.num, self.num), dtype=float)
        self.w_gen = np.ones(self.num, dtype=float)

        self.has_por = np.zeros(self.num, dtype=bool)
        self.has_ret = np.zeros(self.num, dtype=bool)

        self.a_sub = np.zeros(self.num, dtype=float)
        self.a_sur = np.zeros(self.num, dtype=float)
        self.a_por = np.zeros(self.num, dtype=float)
        self.a_ret = np.zeros(self.num, dtype=float)
        self.a_gen = np.zeros(self.num, dtype=float)

    def add_sur_link(self, src: int, dst: int, weight: float = 1.0):
        self.w_sur[src, dst] = weight
        
    def add_sub_link(self, src: int, dst: int, weight: float = 1.0):
        self.w_sub[src, dst] = weight

    def add_por_link(self, src: int, dst: int, weight: float = 1.0):
        self.w_por[src, dst] = weight
        self.has_por[dst] = True

    def add_ret_link(self, src: int, dst: int, weight: float = 1.0):
        self.w_ret[src, dst] = weight
        self.has_ret[dst] = True
            
    def set_gen_loop(self, i: int, weight: float = 1.0):
        self.w_gen[i] = weight

    def reserve(self, num: int):            
        if num <= self.num: 
            return
        
        if self.num == 0:
            self.w_sub = np.zeros((num, num), dtype=float)
            self.w_sur = np.zeros((num, num), dtype=float)
            self.w_por = np.zeros((num, num), dtype=float)
            self.w_ret = np.zeros((num, num), dtype=float)
            
            self.w_gen = np.ones(num, dtype=float)
            
            self.has_por = np.zeros(num, dtype=bool)
            self.has_ret = np.zeros(num, dtype=bool)
            
            self.a_sub = np.zeros(num, dtype=float)
            self.a_sur = np.zeros(num, dtype=float)
            self.a_por = np.zeros(num, dtype=float)
            self.a_ret = np.zeros(num, dtype=float)
            
            self.a_gen = np.zeros(num, dtype=float)
            
            self.num = num
            return

        def grow(existing_weights):
            new_weights = np.zeros((num, num), dtype=float)
            new_weights[:self.num,:self.num] = existing_weights
            return new_weights
        
        self.w_sub = grow(self.w_sub)
        self.w_sur = grow(self.w_sur)
        self.w_por = grow(self.w_por)
        self.w_ret = grow(self.w_ret)
        
        self.w_gen = np.pad(self.w_gen, (0, num-self.num), constant_values=1.0)
        
        self.has_por = np.pad(self.has_por, (0, num-self.num))
        self.has_ret = np.pad(self.has_ret, (0, num-self.num))
        
        self.a_sub = np.pad(self.a_sub, (0, num-self.num))
        self.a_sur = np.pad(self.a_sur, (0, num-self.num))
        self.a_por = np.pad(self.a_por, (0, num-self.num))
        self.a_ret = np.pad(self.a_ret, (0, num-self.num))
        self.a_gen = np.pad(self.a_gen, (0, num-self.num))
        self.num = num

    def _propagate(self):
        z_gen = self.a_gen * self.w_gen
        z_por = self.w_por.T @ self.a_por
        z_ret = self.w_ret.T @ self.a_ret
        z_sub = self.w_sub.T @ self.a_sub
        z_sur = self.w_sur.T @ self.a_sur
        return z_gen, z_por, z_ret, z_sub, z_sur
    
    def calculate(self, z_gen, z_por, z_ret, z_sub, z_sur):
        # gen
        cond_gen_reset = (z_gen * z_sub == 0) | (self.has_por & (z_por == 0))
        a_gen_next = np.where(cond_gen_reset, z_sur, z_gen * z_sub)

        # por
        cond_por_off = (z_sub <= 0) | (self.has_por & (z_por <= 0))
        a_por_next = np.where(cond_por_off, 0, z_sur + z_gen)

        # ret
        a_ret_next = np.where(z_por < 0, 1.0, 0.0)

        # sub
        cond_sub_off = (z_gen != 0) | (self.has_por & (z_por <= 0))
        a_sub_next = np.where(cond_sub_off, 0.0, z_sub)

        # sur
        cond_sur_zero = (z_sub <= 0) | (self.has_por & (z_por <= 0))
        sur_body = z_sur + z_gen
        sur_with_ret = sur_body * z_ret
        a_sur_next = np.where(
            cond_sur_zero, 0.0,
            np.where(self.has_ret, sur_with_ret, sur_body)
        )
        return a_gen_next, a_por_next, a_ret_next, a_sub_next, a_sur_next

    def _calculate(self, z_gen, z_por, z_ret, z_sub, z_sur):
        # 1) REQUEST persists unless inhibited by POR  (← inhibit-request)
        #    If any incoming POR is negative, the unit is suppressed and cannot hold/receive a request.
        a_sub_next = np.where(z_por < 0, 0.0, np.maximum(self.a_sub, z_sub))
        requested = (a_sub_next > 0)

        # 2) SUR only travels upward while requested AND not inhibited by RET  (← inhibit-confirm)
        #    If incoming RET > 0 (from successors), parent must not confirm yet.
        sur_drive = z_gen + z_sur
        a_sur_next = np.where(requested & (z_ret <= 0), sur_drive, 0.0)

        # 3) POR is sent to successors while the unit is requested but NOT yet true;
        #    turn POR OFF once a child has confirmed (i.e., we receive positive SUR from children).
        #    (Table 1: R/A/W/F send inhibit-request; T/C do not.) 
        is_true = (z_sur > 0)        # simple numeric surrogate for "child confirmed"
        a_por_next = np.where(requested & (~is_true), -1.0, 0.0)

        # 4) RET is sent to predecessors whenever the unit participates in a sequence
        #    (i.e., has a successor) and is still requested; last-in-sequence has no successor => no RET.
        a_ret_next = np.where(requested & self.has_ret, 1.0, 0.0)

        # 5) GEN: keep your associator/pass-through, but only meaningful while requested.
        a_gen_next = np.where(requested, np.where((z_gen * z_sub == 0) | (self.has_por & (z_por == 0)), z_sur, z_gen * z_sub), 0.0)
                
        return a_gen_next, a_por_next, a_ret_next, a_sub_next, a_sur_next
    
    def _clip(self, a_gen, a_por, a_ret, a_sub, a_sur):
        # Quantize/clamp as suggested in the paper:
        # por/ret ∈ {−1,0,1}; sub ∈ {0,1}; sur ∈ [0,1] or −1 (solid fail); gen left as-is (clipped)
        eps = 1e-9

        # por: sign to {-1,0,1}
        ap = np.zeros_like(a_por)
        ap[a_por > eps] = 1.0
        ap[a_por < -eps] = -1.0
        a_por = ap

        # ret: strictly 0/1
        a_ret = (a_ret > 0).astype(float)

        # sub: strictly 0/1 (requests)
        a_sub = (a_sub > 0).astype(float)

        # sur: clip to [-1, 1] (−1 means solid fail)
        a_sur = np.clip(a_sur, -1.0, 1.0)

        # gen: keep finite, softly clipped
        a_gen = np.clip(a_gen, -1.0, 1.0)

        return a_gen, a_por, a_ret, a_sub, a_sur

    def step(self):
        z_gen, z_por, z_ret, z_sub, z_sur = self._propagate()
        a_gen_next, a_por_next, a_ret_next, a_sub_next, a_sur_next = self.calculate(z_gen, z_por, z_ret, z_sub, z_sur)

        a_gen_next, a_por_next, a_ret_next, a_sub_next, a_sur_next = self._clip(a_gen_next, a_por_next, a_ret_next, a_sub_next, a_sur_next)

        self.a_gen = a_gen_next
        self.a_por = a_por_next
        self.a_ret = a_ret_next
        self.a_sub = a_sub_next
        self.a_sur = a_sur_next

    def request(self, node: int, value: float = 1.0):
        self.a_sub[node] = value  # will propagate on next step
        
    def reset(self):
        self.a_gen.fill(0)
        self.a_por.fill(0)
        self.a_ret.fill(0)
        self.a_sub.fill(0)
        self.a_sur.fill(0)

    # Query
    def confirmed(self, node: int, threshold: float = 0.5) -> bool:
        # return (self.a_gen[node] > threshold).all()
        return (self.a_sur[node] > threshold).all() or (self.a_gen[node] > threshold).all()

        # return (self.a_sur[node] > threshold).all()

    def failed(self, node: int) -> bool:
        return (self.a_sur[node] < 0)
    
    def confirmed_list(self, nodes: list[int], threshold: float = 0.5) -> list[int]:
        sur = self.a_sur[nodes]   # 1-D
        gen = self.a_gen[nodes]   # 1-D
        mask = ((sur > threshold) | (gen > threshold))  # 1-D boolean
        return [n for n, m in zip(nodes, mask) if m]

    def por_successors(self, node: int, weight_threshold: float = 0) -> list[int]:
        return np.flatnonzero(self.w_por[node] > weight_threshold).tolist()
