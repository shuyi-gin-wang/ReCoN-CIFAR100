import torch
import numpy as np
from recon import ReCoN


def predict_image(image, model):
    with torch.no_grad():
        out = model(image)
        predicted_class = out["class"].argmax(1).item()
        predicted_superclass = out["superclass"].argmax(1).item()
    return predicted_class, predicted_superclass


class CIFARReCoNBuilder:
    def __init__(self, net: ReCoN):
        self.net = net
        self.superclass_terminal_netid_mapping = {}
        self.class_terminal_netid_mapping = {}
        self.superclass_netids = []
        
    def build(self, image, model, max_steps = 10):
        # model evaluates image
        predicted_class, predicted_superclass = predict_image(image, model)
        
        if self.net.num == 1:
            _, superclass_netid, superclass_verifier_netid, _, _, class_verifier_netid = self._add_basic_super_unit(0)
            self.class_terminal_netid_mapping[class_verifier_netid] = predicted_class
            self.superclass_terminal_netid_mapping[superclass_verifier_netid] = predicted_superclass
            self.superclass_netids.append(superclass_netid)
            print(f"+++ Added new super unit with class {predicted_class} and predict superclass {predicted_superclass} (superclass netid: {superclass_netid})")
            return 
                        
        superclass_terminal_netids = np.array(list(self.superclass_terminal_netid_mapping.keys()))
        class_terminal_netids = np.array(list(self.class_terminal_netid_mapping.keys()))

        superclass_mask = np.array([self.superclass_terminal_netid_mapping[i] == predicted_superclass for i in superclass_terminal_netids])
        class_mask = np.array([self.class_terminal_netid_mapping[i] == predicted_class for i in class_terminal_netids])
                
        self.net.reset()

        confirmed_superclasses = []
        for _ in range(max_steps):
            self.net.request(0, 1.0)
            self.net.step()
                
            self.net.a_sur[superclass_terminal_netids] = 0
            self.net.a_sur[class_terminal_netids] = 0

            self.net.a_sur[superclass_terminal_netids[superclass_mask]] = 1
            self.net.a_sur[class_terminal_netids[class_mask]] = 1

            if self.net.confirmed(0):
                print(f">>> CONFIRMED with class {predicted_class} under predict superclass {predicted_superclass} with steps {_}")
                return
            
            confirmed_superclass = self.net.confirmed_list(self.superclass_netids)
            if confirmed_superclass: confirmed_superclasses = confirmed_superclass

        assert(len(confirmed_superclasses) <= 1) 
        
        if confirmed_superclasses:
            # check if any super class confirmations 
            por_successors = self.net.por_successors(confirmed_superclasses[0]) # class hypo netid
            assert(len(por_successors) == 1) # there should only be one superclass confirmed and one associated class hypo
            _, class_verifier_netid = self._add_basic_class_unit(por_successors[0])
            self.class_terminal_netid_mapping[class_verifier_netid] = predicted_class
            print(f"+ Added new class unit with class {predicted_class} under predict superclass {predicted_superclass} (superclass netid: {confirmed_superclasses[0]})")
        else:
            # add new superunit 
            _, superclass_netid, superclass_verifier_netid, _, _, class_verifier_netid = self._add_basic_super_unit(0)
            self.class_terminal_netid_mapping[class_verifier_netid] = predicted_class
            self.superclass_terminal_netid_mapping[superclass_verifier_netid] = predicted_superclass
            self.superclass_netids.append(superclass_netid)
            print(f"+++ Added new super unit with class {predicted_class} and predict superclass {predicted_superclass} (superclass netid: {superclass_netid})")

    def _add_basic_super_unit(self, root_id):
        unit_netid, superclass_netid, superclass_verifier_netid, class_hypo_netid, class_netid, class_verifier_netid = list(range(self.net.num, self.net.num + 6))  
        self.net.reserve(self.net.num + 6)

        self.net.add_sub_link(root_id, unit_netid)
        self.net.add_sur_link(unit_netid, root_id)
        
        self.net.add_sub_link(unit_netid, superclass_netid)
        self.net.add_sub_link(unit_netid, class_hypo_netid)
        self.net.add_sur_link(superclass_netid, unit_netid)
        self.net.add_sur_link(class_hypo_netid, unit_netid)
        
        self.net.add_por_link(superclass_netid, class_hypo_netid)
        self.net.add_ret_link(class_hypo_netid, superclass_netid)
        
        self.net.add_sur_link(superclass_verifier_netid, superclass_netid)
        
        self.net.add_sub_link(class_hypo_netid, class_netid)
        self.net.add_sur_link(class_netid, class_hypo_netid)
        
        self.net.add_sur_link(class_verifier_netid, class_netid)
        
        return unit_netid, superclass_netid, superclass_verifier_netid, class_hypo_netid, class_netid, class_verifier_netid
        
    def _add_basic_class_unit(self, root_id):
        class_netid, class_verifier_netid = list(range(self.net.num, self.net.num + 2))  
        self.net.reserve(self.net.num + 2)

        self.net.add_sub_link(root_id, class_netid)
        self.net.add_sur_link(class_netid, root_id)
        
        self.net.add_sur_link(class_verifier_netid, class_netid)
        
        return class_netid, class_verifier_netid
