import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, Dropout
import torch_geometric.transforms as T
from torch_geometric.nn import EdgeConv
from torch_geometric.nn import knn_graph

def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

class Net(torch.nn.Module):
    def __init__(self, out_channels=10, k=10, aggr='max', dilation=4):
        super(Net, self).__init__()

        self.k = k
        self.dilation = dilation
        
        #pos
        self.conv1 = EdgeConv(MLP([3*2, 64, 64]), aggr)
        self.conv2 = EdgeConv(MLP([3*2, 64, 64]), aggr)
        #x
        self.conv3 = EdgeConv(MLP([25*2, 64, 64]), aggr)
        self.conv4 = EdgeConv(MLP([25*2, 64, 64]), aggr)

        self.lin1 = MLP([4 * 64, 4 * 64, 1024])
        
        # FIXME:
        self.mlp = Seq(MLP([1024 + 64 * 4 + 28 , 256]), Dropout(0.5), MLP([256, 128]),
                       Dropout(0.5), Lin(128, out_channels))

        self.mano_forward = Seq(MLP([778, 256]), Dropout(0.5), MLP([256, 128]), Dropout(0.5), MLP([128, 64]), Dropout(0.5), Lin(64, 15))

        ## !!
    def forward(self, x, pos, onehot, batch):

        edge_index_pos = knn_graph(pos, k=self.k, batch=batch, loop=True)
        edge_index_x = knn_graph(x, k=self.k, batch=batch, loop=True)

        dilated_ei_pos = edge_index_pos[:, ::self.dilation]
        dilated_ei_x = edge_index_x[:, ::self.dilation]

        x1 = self.conv1(pos, edge_index_pos)
        x2 = self.conv2(pos, dilated_ei_pos)
        x3 = self.conv3(x, edge_index_x)
        x4 = self.conv4(x, dilated_ei_x)

        feat = torch.cat([x1, x2, x3, x4], dim=1)

        out = self.lin1(feat)

        #feat = feat.view(-1,2826,1024) # 778 + 2048

        global_feature = out.view(-1,2826,1024)

        global_hand_max = torch.max(global_feature,2,keepdim=True)[0][:,:778,0]

        mano_param = self.mano_forward(global_hand_max)

        #[32, 2826, 1024]
        global_feature_max = torch.max(global_feature,1,keepdim=True)[0]

        global_feature_max = global_feature_max.expand(global_feature.shape[0],2826,1024)

        # [32, 2826, 1024]

        feat = feat.view(-1,2826,64 * 4)

        # [32, 2826, 256]
        # FIXME:
        onehot = onehot.view(-1,2826,28) ## !!

        # [32, 2826, 28]
        # import pdb; pdb.set_trace()

        out = torch.cat([global_feature_max,onehot,feat],dim=2)

        # FIXME:
        out = out.view(-1, 1024 + 64 * 4 + 28) ## !!

        ### Point net part_segmentation method

        out = self.mlp(out)

        # F.sigmoid(out)
        
        return F.log_softmax(out, dim=1), mano_param
        