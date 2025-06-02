# %%
import numpy as np
import pandas as pd
from collections import defaultdict
import random

'''
data object to hold probe dataframe as class object to avoid duplicating work when splitting states
'''
class Probe:
    def __init__(self):
        pass

    def init_df(self, probe_df):
        self.state_names = []
        self.state_dfs = []

        for (state_name,), state_df in probe_df.groupby(by=["labels"]):
            self.state_names.append(state_name)
            self.state_dfs.append(state_df)
        return self

    def init_states(self, state_names, state_dfs):
        self.state_names = state_names
        self.state_dfs = state_dfs
        return self

    def build_df(self):
        return pd.concat(self.state_dfs, axis=0)
    

def warp(seq, adjust):
    original_index = np.arange(len(seq))
    new_index = np.linspace(0, len(seq) - 1, int(len(seq) * adjust))
    
    warped_series = pd.Series(np.interp(new_index, original_index, seq.values), index=new_index)
    return warped_series


class DataAugmentor:
    def __init__(self, probes, transition_matrix=None, voltage_column="pre_rect", starting_state="NP", ending_state="NP"):
        # support to instantiate without transition_matrix provided
        if transition_matrix is not None:
            self.transition_matrix = transition_matrix
            self.states = transition_matrix.keys()
            self.starting_state = starting_state
            self.ending_state = ending_state
        self.voltage_column = voltage_column

        self.data_by_state = defaultdict(list)
        for probe in probes:
            for state_name, state_df in zip(probe.state_names, probe.state_dfs):
                self.data_by_state[state_name].append(state_df)
            
        self.probes = probes
        '''
        noise_freq
        merge_shift_voltage
        merge_shift_freq
        other operations from torchvision
        '''

    def augment_mixup(self, probe=None, alpha=0.2, add_to_probe_list=False):
        if probe is None:
            probe = random.choice(self.probes)
        other = random.choice(self.probes).build_df()
        own = probe.build_df()

        lamba = np.random.beta(alpha, alpha)
        mixed_series = lamba * own[self.voltage_column]  + (1 - lamba) * other[self.voltage_column]
        mixed_label = lamba * own["labels"] + (1 - lamba) * other["labels"]

        synthetic_probe = Probe().init_df(mixed_series, mixed_label)
        if add_to_probe_list:
            self.probes.append(synthetic_probe)
        return synthetic_probe


    def augment_concat_self(self, p, probe=None, add_to_probe_list=False):
        '''
        given a single probe, for each state with probability p, concat the same seq to itself
        '''
        if probe is None:
            probe = random.choice(self.probes)
        new_state_dfs = [state_df.copy() for state_df in probe.state_dfs]
        for i in range(len(probe.state_dfs)):
            if random.random() <= p:
                new_state_dfs[i] = pd.concat([new_state_dfs[i], new_state_dfs[i]], axis=0).copy()
        
        synthetic_probe = Probe().init_states(probe.state_names, new_state_dfs)
        if add_to_probe_list:
            self.probes.append(synthetic_probe)
        return synthetic_probe

    def augment_concat_other(self, p, probe=None, add_to_probe_list=False):
        '''
        given a single probe, for each state with probability p, concat another seq of the same type
        '''
        if probe is None:
            probe = random.choice(self.probes)
        new_state_dfs = [state_df.copy() for state_df in probe.state_dfs]
        for i in range(len(probe.state_dfs)):
            if random.random() <= p:
                state_name = probe.state_names[i]
                new_seq = random.choice(self.data_by_state[state_name])
                new_state_dfs[i] = pd.concat([new_state_dfs[i],new_seq], axis=0).copy()
        
        synthetic_probe = Probe().init_states(probe.state_names, new_state_dfs)
        if add_to_probe_list:
            self.probes.append(synthetic_probe)
        return synthetic_probe

    def augment_warp_overall(self, probe=None, lower=0.25, upper=4, add_to_probe_list=False):
        '''
        given a single probe, warp the time in each state by a factor of U[lower, upper]
        '''
        if probe is None:
            probe = random.choice(self.probes)
        adjust = np.random.uniform(lower, upper)
        new_state_dfs = []
        for state_df in probe.state_dfs:
            new_state_df = state_df.copy()
            new_state_df[self.voltage_column] = warp(new_state_df[self.voltage_column], adjust)
            new_state_dfs.append(new_state_df)
        
        synthetic_probe = Probe().init_states(probe.state_names, new_state_dfs)
        if add_to_probe_list:
            self.probes.append(synthetic_probe)
        return synthetic_probe
    
    def augment_warp_by_state(self, p, probe=None, lower=0.25, upper=4, add_to_probe_list=False):
        '''
        given a single probe, for each state with probability p, warp the time by a factor of U[lower, upper]
        '''
        if probe is None:
            probe = random.choice(self.probes)
        new_state_dfs = [state_df.copy() for state_df in probe.state_dfs]
        for i in range(len(probe.state_dfs)):
            if random.random() <= p:
                adjust = np.random.uniform(lower, upper)
                new_state_dfs[i][self.voltage_column] = warp(new_state_dfs[i][self.voltage_column], adjust)
        
        synthetic_probe = Probe().init_states(probe.state_names, new_state_dfs)
        if add_to_probe_list:
            self.probes.append(synthetic_probe)
        return synthetic_probe

    def augment_change_amplitude_overall(self, probe=None, lower=0.05, upper=20, add_to_probe_list=False):
        '''
        given a single probe, shift the amplitude by U[lower, upper]
        '''
        if probe is None:
            probe = random.choice(self.probes)
        adjust = np.random.uniform(lower, upper)
        new_state_dfs = []
        for state_df in probe.state_dfs:
            new_state_df = state_df.copy()
            new_state_df[self.voltage_column] *= adjust
            new_state_dfs.append(new_state_df)
        
        synthetic_probe = Probe().init_states(probe.state_names, new_state_dfs)
        if add_to_probe_list:
            self.probes.append(synthetic_probe)
        return synthetic_probe
    
    def augment_change_amplitude_by_state(self, p, probe=None, lower=0.05, upper=20, add_to_probe_list=False):
        '''
        given a single probe, for each state with probability p shift the amplitude by U[lower, upper]
        '''
        if probe is None:
            probe = random.choice(self.probes)
        new_state_dfs = [state_df.copy() for state_df in probe.state_dfs]
        for i in range(len(probe.state_dfs)):
            if random.random() <= p:
                adjust = np.random.uniform(lower, upper)
                new_state_dfs[i][self.voltage_column] *= adjust
        
        synthetic_probe = Probe().init_states(probe.state_names, new_state_dfs)
        if add_to_probe_list:
            self.probes.append(synthetic_probe)
        return synthetic_probe
    

    def augment_noise_voltage_overall(self, stdev, probe=None, add_to_probe_list=False):
        '''
        given a single probe, add gaussian noise with given stdev to the whole sequence 
        '''
        if probe is None:
            probe = random.choice(self.probes)
        new_state_dfs = []
        for state_df in probe.state_dfs:
            new_state_df = state_df.copy()
            noise = np.random.normal(0, stdev, size=new_state_df.shape[0])
            new_state_df[self.voltage_column] += noise
            new_state_dfs.append(new_state_df)

        synthetic_probe = Probe().init_states(probe.state_names, new_state_dfs)
        if add_to_probe_list:
            self.probes.append(synthetic_probe)
        return synthetic_probe

    def augment_noise_voltage_by_state(self, p, stdev, probe=None, add_to_probe_list=False):
        '''
        given a single probe, for each state with probability p, add gaussian noise to each sequence with with given stdev
        '''
        if probe is None:
            probe = random.choice(self.probes)
        new_state_dfs = [state_df.copy() for state_df in probe.state_dfs]
        for i in range(len(probe.state_dfs)):
            if random.random() <= p:
                noise = np.random.normal(0, stdev, size=new_state_dfs[i].shape)
                new_state_dfs[i][self.voltage_column] += noise
        
        synthetic_probe = Probe().init_states(probe.state_names, new_state_dfs)
        if add_to_probe_list:
            self.probes.append(synthetic_probe)
        return synthetic_probe

    def augment_franken(self, add_to_probe_list=False):
        '''
        create new sequence by sampling the next state and sampling matching sequences of that state from observed data
        requires transition matrix to be instantiated in the constructor
        '''
        state_name = self.starting_state
        iterations = 0
        state_dfs = []
        state_names = []
        while (state_name != self.ending_state) or (iterations ==0):
            state_name = np.random.choice(self.states, 1, p=self.transition_matrix.T[state_name])[0]
            
            if len(self.data_by_state[state_name]) == 0:
                # quick check that the code as intended should only be missing NPs
                assert state_name=="NP"
                # if we are missing, then just skip when constructing the new probe
                state_seq = None
            else:
                state_seq = random.choice(self.data_by_state[state_name])
            
            
            if state_seq is not None:
                state_dfs.append(state_seq)
                state_names.append(state_name)

            iterations += 1
        synthetic_probe = Probe().init_states(state_names, state_dfs)
        if add_to_probe_list:
            self.probes.append(synthetic_probe)
        return synthetic_probe

# %%
from transition_matrix import transition_matrix_dict
transition_matrix_df = pd.DataFrame(transition_matrix_dict)
def build_augmented_dataset(all_probes):
    probes_as_cls = [Probe().init_df(p) for p in all_probes]
    DA = DataAugmentor(probes_as_cls, transition_matrix_df)

    p=0.1
    stdev = 0.001
    new_probes = []
    for _ in range(50):
        sample = DA.augment_franken()
        new_probes.append(sample)

    for _ in range(10):
        sample = DA.augment_concat_self(p=p)
        new_probes.append(sample)

    for _ in range(10):
        sample = DA.augment_concat_other(p=p)
        new_probes.append(sample)

    for _ in range(5):
        sample = DA.augment_warp_overall()
        new_probes.append(sample)

    for _ in range(5):
        sample = DA.augment_warp_by_state(p=p)
        new_probes.append(sample)

    for _ in range(10):
        sample = DA.augment_change_amplitude_overall()
        new_probes.append(sample)

    for _ in range(10):
        sample = DA.augment_change_amplitude_by_state(p=p)
        new_probes.append(sample)

    full_DA = DataAugmentor(probes_as_cls+new_probes, transition_matrix=transition_matrix_df)

    for _ in range(50):
        sample = full_DA.augment_franken()
        new_probes.append(sample)

    augmented_probes = probes_as_cls+new_probes
    return [p.build_df() for p in augmented_probes]

# for _ in range(5):
#     sample = DA.augment_noise_voltage_overall(stdev=stdev)
#     new_probes.append(sample)

# for _ in range(5):
#     sample = DA.augment_noise_voltage_by_state(stdev=stdev, p=p)
#     new_probes.append(sample)


# %%
if __name__ == "__main__":
    from utils import split_into_probes_data_leak, load_files
    ROOT = "c:\\Users\\milok\\clasp_transitions"
    transition_matrix_df = pd.DataFrame(transition_matrix_dict)
    test_files = ['cxtonhand30nov2021no1.csv',
    'cxtonhand21oct2021no4.csv',
    'cxtonhand18nov2021no4.csv',
    'cxtonhand10nov2021no7.csv',
    'cxtonhand19nov2021no5.csv',
    'handcxt23aug2021no8.csv',
    'cxtonhand19nov2021no2b.csv',
    'cxtonhand19nov2021no7b.csv',
    'cxtonhand10nov2021no11.csv',
    'cxtonhandkscage9mar2022no7.csv',
    'handcxt3aug2021.csv',
    'cxtonhand24nov2021no9.csv',
    'cxtonhand21sep2021no8.csv']

    big_test_df, test_dfs = load_files(test_files, root=ROOT)

    all_probes, all_probe_names = split_into_probes_data_leak(test_dfs)
    aug_dataset = build_augmented_dataset(all_probes)