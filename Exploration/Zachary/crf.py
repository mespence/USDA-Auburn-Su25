import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

class CRF(nn.Module):
        def __init__(self, state_to_ix, start_state, emission_function):
                super(CRF, self).__init__()
                self.num_states = len(state_to_ix)
                self.state_to_ix = state_to_ix
                self.start_state = start_state
                self.emission_function = emission_function
                
                # Transitions are learned and thus initially random
                self.transitions = nn.Parameter(torch.randn(
                                  self.num_states, self.num_states))
                # self.transitions = torch.eye(self.num_states, self.num_states)

        def log_sum_exp(self, vec):
                # used for softmax
                max_score = vec[0, torch.argmax(vec)]
                max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
                return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

        def _forward_alg(self, emissions):
                print("forward start")
                """
                Compute the partition function for the log likelihood. We do this so that
                we can adjust the transitions to better match our data. Can essentially
                think of this as calculating the likelihood of having the states we do
                and the transitions we do for each time step, all done in log space
                """
                # This is done in log space, hence the -10000.
                init_alphas = torch.full((self.num_states, self.num_states), -10000.)
                init_alphas[:, self.state_to_ix[self.start_state]] = 0
                # wrapping this in a variable makes autograd work
                forward_var = init_alphas
                
                # Now we step over the time steps and then each state
                for emission in emissions.squeeze(0).T:
                        # Calculates the below in a more optimized way
                        emit_score = emission.expand(self.num_states, -1)
                        trans_score = self.transitions
                        next_state_var = forward_var + emit_score + trans_score
                        forward_var = torch.logsumexp(next_state_var, 1, True).view(1, -1).expand(self.num_states, -1)
                        """
                        current_alphas = []
                        for next_state in range(self.num_states):
                                # turn the emission score into a column tensor
                                emit_score = emission[0][next_state].view(1, -1).expand(1, self.num_states)
                                trans_score = self.transitions[next_state].view(1, -1)
                                # get all the potential ways we could have gotten to the current state
                                next_state_var = forward_var + trans_score + emit_score
                                # save the softmax, which we use so we can differentiate everything
                                current_alphas.append(self.log_sum_exp(next_state_var).view(1))
                        forward_var = torch.cat(current_alphas).view(1, -1)
                        """
                final_alpha = self.log_sum_exp(forward_var[0, :].view(1, -1))
                print("forward end")
                return final_alpha
                

        def _viterbi_decode(self, emissions):
                print("viterbi start")
                """
                emissions: the outputs from the previous layer with the same dimension
                           as the number of states.
                returns:
                        the best sequence of states (as indices) and the score
                        of the best path, if that is of interest
                """
                breadcrumbs = []

                # This is done in log space, hence the -10000.
                init_vvars = torch.full((self.num_states, self.num_states), -10000.)
                init_vvars[:, self.state_to_ix[self.start_state]] = 0

                # All that matters to us are the viterbi vars from the last row
                last_vvars = init_vvars
                # Now we step over the time steps and then each state
                for emission in emissions.squeeze(0).T:
                        # Do the below more efficiently
                        emit_score = emission.expand(self.num_states, -1)
                        trans_score = self.transitions
                        next_state_vvar = last_vvars + trans_score + emit_score
                        last_vvars = torch.max(next_state_vvar, dim = 1)[0].view(1, -1).expand(self.num_states, -1)
                        best_next_state = torch.argmax(next_state_vvar, dim = 1).flatten()
                        breadcrumbs.append(best_next_state)
                        """
                        current_vvars = []
                        current_breadcrumbs = []

                        for next_state in range(self.num_states):
                                # The vvalue for this state is the max of starting at any of
                                # the last states and coming to this state from them.
                                # The emmision value for this state is the same regardless
                                # of how we got here so we can add it in later.
                                next_state_vvar = last_vvars + self.transitions[next_state]
                                best_next_state = torch.argmax(next_state_vvar)
                                # .view(1) just flattens the single-dimensional tensor
                                current_vvars.append(next_state_vvar[0][best_next_state].view(1))
                                current_breadcrumbs.append(best_next_state)
                        # Here we add in the emmisions, view makes it a column tensor
                        """
                
                # Finish it off by returning the final best score along with the state sequence
                best_state = torch.argmax(last_vvars)
                path_score = last_vvars[0][best_state]
                best_path = [best_state]
                for current_breadcrumbs in reversed(breadcrumbs):
                        best_state = current_breadcrumbs[best_state]
                        best_path.append(best_state)
                # Remove the first one since it is where we assumed we started, and therefore
                # our list is 1 too long
                best_path.pop()
                best_path.reverse()
                print("viterbi end")
                return best_path, path_score

        def _score_input(self, emissions, states):
                print("score start")
                score = torch.zeros(1)
                states = torch.cat([torch.tensor([self.state_to_ix[self.start_state]]), states.squeeze(0)])
                for i, emission in enumerate(emissions.squeeze(0).T):
                        score += self.transitions[states[i + 1], states[i]] + \
                                 emission.squeeze()[states[i + 1]]
                print("score end")
                return score

        def neg_log_likelihood(self, inputs, labels):
                emissions = self.emission_function(inputs)
                forward_score = self._forward_alg(emissions)
                gold_score = self._score_input(emissions, labels)
                return forward_score - gold_score

        def forward(self, waveform):
                emissions = self.emission_function(waveform)
                label_sequence, score = self._viterbi_decode(emissions)
                return label_sequence, score

state_to_ix = {
        'NP' : 0,
        'J'  : 1,
        'K'  : 2,
        'L'  : 3,
        'M'  : 4,
        'N'  : 5,
        'Z'  : 6,
}

ix_to_state = {
    v: k for k, v in state_to_ix.items()
}

# Example Instantiation
# model = CRF(state_to_ix, 'NP', lambda _ : test_emission_1)

def train_unet(epochs = 80, lr = 5e-4):
    import unet # Milo's UNET
    unet_mod = unet.UNet1D(in_channels = 1, out_channels = len(state_to_ix))
    crf_unet = CRF(state_to_ix, 'NP', unet_mod)
    crf_unet.to(device)
    optimizer = optim.Adam(crf_unet.parameters(), lr = lr, capturable=False)
    for epoch in range(epochs):
        crf_unet.train()
        running_loss = 0.0
        for batch in tqdm(unet.tr_dataloader):
            x, y, _ = batch
            x, y = x.unsqueeze(0).to(device), \
                   y.unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            outputs = crf_unet(x.unsqueeze(1))
            
            loss = crf_unet.neg_log_likelihood(x.unsqueeze(1), y)
            print(crf_unet.transitions)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(unet.tr_dataloader)
        acc, f1 = unet.evaluate(crf_unet, unet.test_dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}')
        print(crf_unet.transitions.data)

train_unet()
