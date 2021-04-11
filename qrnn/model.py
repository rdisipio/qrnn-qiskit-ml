import torch.nn as nn
import torch

from qiskit  import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.opflow import AerPauliExpectation
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import CircuitQNN, TwoLayerQNN
from qiskit_machine_learning.connectors import TorchConnector


class QLSTM(nn.Module):
    def __init__(self, 
                input_size: int, 
                hidden_size: int, 
                n_qubits: int=4,
                n_qlayers: int=1,
                batch_first=True,
                backend='statevector_simulator'):
        super(QLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.batch_first = batch_first
        
        self.clayer_in = nn.Linear(self.concat_size, n_qubits)
        self.clayer_out = nn.Linear(self.n_qubits, self.hidden_size)

        self.qi = QuantumInstance(Aer.get_backend('statevector_simulator'))
        feature_map = ZZFeatureMap(self.n_qubits)
        ansatz = RealAmplitudes(self.n_qubits, reps=self.n_qlayers)

        self.qnn1 = TwoLayerQNN(self.n_qubits, feature_map, ansatz, exp_val=AerPauliExpectation(), quantum_instance=self.qi)
        self.qnn2 = TwoLayerQNN(self.n_qubits, feature_map, ansatz, exp_val=AerPauliExpectation(), quantum_instance=self.qi)
        self.qnn3 = TwoLayerQNN(self.n_qubits, feature_map, ansatz, exp_val=AerPauliExpectation(), quantum_instance=self.qi)
        self.qnn4 = TwoLayerQNN(self.n_qubits, feature_map, ansatz, exp_val=AerPauliExpectation(), quantum_instance=self.qi)

        self.qlayer = {
            'forget': TorchConnector(self.qnn1),
            'input': TorchConnector(self.qnn2),
            'update': TorchConnector(self.qnn3),
            'output': TorchConnector(self.qnn4)
        }

    def forward(self, x, init_states=None):
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()
        
        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            print(">>>", y_t.size(), self.qlayer['forget'](y_t).size())

            f_t = torch.sigmoid(self.clayer_out(self.qlayer['forget'](y_t)))  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.qlayer['input'](y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.qlayer['update'](y_t)))  # update block
            o_t = torch.sigmoid(self.clayer_out(self.qlayer['output'](y_t))) # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        return hidden_seq, (h_t, c_t)

