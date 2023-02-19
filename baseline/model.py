import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self, args):
        super(BasicModel, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.n_user, self.m_item = args.user_num, args.item_num
        self.dim = args.dim
        self.user_embedding = nn.Embedding(self.n_user, self.dim)
        self.item_embedding = nn.Embedding(self.m_item, self.dim)
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)


class VanillaMF(BasicModel):
    def __init__(self, args):
        super(VanillaMF, self).__init__(args)
        self.act = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

    def forward(self, instances):
        users_emb = self.user_embedding(instances.long())
        items_emb = self.item_embedding.weight
        ratings = self.act(torch.matmul(users_emb, items_emb.t()))
        return ratings

    def bpr_loss(self, users, pos_items, neg_items):
        users_emb = self.user_embedding(users.long())
        pos_items_emb = self.item_embedding(pos_items.long())
        neg_items_emb = self.item_embedding(neg_items.long())
        pos_ratings = torch.sum(users_emb * pos_items_emb, dim=1)
        neg_ratings = torch.sum(users_emb * neg_items_emb, dim=1)

        loss = torch.mean(F.softplus(neg_ratings - pos_ratings))
        reg_loss = (1/2)*(users_emb.norm(2).pow(2) +
                          pos_items_emb.norm(2).pow(2) +
                          neg_items_emb.norm(2).pow(2))/float(len(users))
        return loss, reg_loss

    def loss_func(self, preds, labels):
        return self.bce_loss(preds, labels)


class NeuMF(BasicModel):
    def __init__(self, args):
        super(NeuMF, self).__init__(args)
        self.act = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()
        self.mlp = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * self.dim, 1 * self.dim), nn.Tanh(),
                                 nn.Dropout(0.2), nn.Linear(1 * self.dim, 1 * self.dim), nn.Tanh())
        self.combined = nn.Sequential(nn.Linear(2 * self.dim, 1), nn.Sigmoid())

    def forward(self, instances):
        users_emb = self.user_embedding(instances.long())
        items_emb = self.item_embedding.weight
        users_emb = users_emb.unsqueeze(1).repeat(1, self.m_item, 1)
        items_emb = items_emb.unsqueeze(0).repeat(instances.shape[0], 1, 1)
        cos = users_emb * items_emb
        mlp_in = torch.cat((users_emb, items_emb), dim=2)
        mlp_out = self.mlp(mlp_in)
        f_in = torch.cat((cos, mlp_out), dim=2)
        f_out = self.combined(f_in)
        ratings = f_out.squeeze()
        return ratings

    def loss_func(self, preds, labels):
        return self.bce_loss(preds, labels)


class RankNet(BasicModel):
    def __init__(self, args):
        super(RankNet, self).__init__(args)
        self.dropout = args.dropout
        self.embed = nn.Sequential(nn.Linear(2 * self.dim, 1 * self.dim), nn.Dropout(0.2),
                                   nn.ReLU(), nn.Linear(1 * self.dim, 1))
        self.act = nn.Sigmoid()
        self.mode = 'train'

    def bpr_loss(self, users, pos_items, neg_items):
        users_emb = self.user_embedding(users.long())
        pos_items_emb = self.item_embedding(pos_items.long())
        neg_items_emb = self.item_embedding(neg_items.long())
        pos_ratings = self.embed(torch.cat([users_emb, pos_items_emb], dim=1))
        neg_ratings = self.embed(torch.cat([users_emb, neg_items_emb], dim=1))

        loss = torch.mean(self.act(neg_ratings - pos_ratings))
        reg_loss = (1/2)*(self.user_embedding(users.long()).norm(2).pow(2) +
                          self.item_embedding(pos_items.long()).norm(2).pow(2) +
                          self.item_embedding(neg_items.long()).norm(2).pow(2))/float(len(users))
        return loss, reg_loss

    def forward(self, instances):
        users_emb = self.user_embedding(instances.long())
        items_emb = self.item_embedding.weight
        users_emb = users_emb.unsqueeze(1).repeat(1, self.m_item, 1)
        items_emb = items_emb.unsqueeze(0).repeat(instances.shape[0], 1, 1)
        emb = torch.cat([users_emb, items_emb], dim=2)
        ratings = self.embed(emb).squeeze()
        return ratings


class BasicCTRModel(nn.Module):
    def __init__(self, args):
        super(BasicCTRModel, self).__init__()
        self.device = args.device
        self.batch_size = args.batch_size
        self.dim = args.dim
        self.feature_name = ['Brand_ID', 'Cate1_ID', 'Cate2_ID', 'Cate3_ID', 'Region_ID']
        self.feature_num = {'Brand_ID': args.user_num,
                            'Cate1_ID': args.cate_num[0],
                            'Cate2_ID': args.cate_num[1],
                            'Cate3_ID': args.cate_num[2],
                            'Region_ID': args.item_num}
        self.feat_dim = len(self.feature_name) * self.dim
        self.field_dim = len(self.feature_name)

        self.feature_layers = nn.ModuleDict({key: nn.Embedding(self.feature_num[key], self.dim)
                                             for key in self.feature_name})
        self.feature_layers_first_order = nn.ModuleDict({key: nn.Embedding(self.feature_num[key], 1)
                                                         for key in self.feature_name})
        self.deep_projection_layers = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim), nn.Tanh(), nn.Dropout(0.2),
                                                    nn.Linear(self.feat_dim, self.feat_dim), nn.Tanh(), nn.Dropout(0.2))

        self.act = nn.Sigmoid()
        self.bce_loss = nn.BCELoss()

        self.output_layer = nn.Sequential(nn.Linear(self.feat_dim, 1), nn.Sigmoid())
        for modules in [self.feature_layers_first_order.modules(),
                        self.feature_layers.modules(),
                        self.deep_projection_layers.modules(),
                        self.output_layer.modules()]:
            for module in modules:
                if isinstance(module, (nn.Embedding, nn.Linear)):
                    nn.init.xavier_normal_(module.weight)

    def generate_batch(self, instances):
        batch_size = instances['Brand_ID'].shape[0]
        instances = {key: instances[key].unsqueeze(1).repeat(1, self.feature_num['Region_ID']).flatten()
                     for key in self.feature_name[:-1]}
        instances['Region_ID'] = torch.LongTensor(list(range(self.feature_num['Region_ID'])))\
            .unsqueeze(0).repeat(batch_size, 1).flatten().to(self.device)
        return instances

    def forward(self, instances):
        instances = self.generate_batch(instances)
        features = {key: self.feature_layers[key](instances[key]) for key in self.feature_name}

        # Deep module
        emb = torch.cat([features[key] for key in self.feature_name], dim=1)
        deep_emb = self.deep_projection_layers(emb)

        ratings = self.output_layer(deep_emb).reshape(-1, self.feature_num['Region_ID'])
        return ratings

    def loss_func(self, preds, labels):
        return self.bce_loss(preds, labels)


class DeepFM(BasicCTRModel):
    def __init__(self, args):
        super(DeepFM, self).__init__(args)
        self.output_layer = nn.Sequential(nn.Linear(self.field_dim + self.dim + self.feat_dim, 1), nn.Sigmoid())

        for module in self.output_layer.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.xavier_normal_(module.weight)

    def forward(self, instances):
        instances = self.generate_batch(instances)
        # Lookup embedding
        features_first_order = {key: self.feature_layers_first_order[key](instances[key])
                                for key in self.feature_name}
        features = {key: self.feature_layers[key](instances[key]) for key in self.feature_name}

        # FM module
        fm_first_order = torch.cat([features_first_order[key] for key in self.feature_name], dim=1)
        fm_second_order_sum_square = sum([features[key] for key in self.feature_name]) ** 2
        fm_second_order_square_sum = sum([features[key] ** 2 for key in self.feature_name])
        fm_second_order = (fm_second_order_sum_square - fm_second_order_square_sum) * 0.5

        # Deep module
        emb = torch.cat([features[key] for key in self.feature_name], dim=1)
        deep_emb = self.deep_projection_layers(emb)

        # Output concat
        output = torch.cat([fm_first_order, fm_second_order, deep_emb], dim=1)

        ratings = self.output_layer(output).reshape(-1, self.feature_num['Region_ID'])
        return ratings


class WideDeep(BasicCTRModel):
    def __init__(self, args):
        super(WideDeep, self).__init__(args)
        self.output_layer = nn.Sequential(nn.Linear(self.field_dim + self.feat_dim, 1), nn.Sigmoid())

        for module in self.output_layer.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.xavier_normal_(module.weight)

    def forward(self, instances):
        instances = self.generate_batch(instances)
        # Lookup embedding
        features_first_order = {key: self.feature_layers_first_order[key](instances[key])
                                for key in self.feature_name}
        features = {key: self.feature_layers[key](instances[key]) for key in self.feature_name}

        # FM module
        fm_first_order = torch.cat([features_first_order[key] for key in self.feature_name], dim=1)

        # Deep module
        emb = torch.cat([features[key] for key in self.feature_name], dim=1)
        deep_emb = self.deep_projection_layers(emb)

        # Output concat
        output = torch.cat([fm_first_order, deep_emb], dim=1)

        ratings = self.output_layer(output).reshape(-1, self.feature_num['Region_ID'])
        return ratings


class xDeepFM(BasicCTRModel):
    def __init__(self, args):
        super(xDeepFM, self).__init__(args)
        self.field_nums = [self.field_dim]
        self.conv1ds = nn.ModuleList()
        size = 100
        for i in range(2):
            self.conv1ds.append(nn.Conv1d(self.field_nums[-1] * self.field_nums[0], size, 1))
            self.field_nums.append(size)

        self.output_layer = nn.Sequential(nn.Linear(self.field_dim + 2 * size + self.feat_dim, 1), nn.Sigmoid())

        for module in self.output_layer.modules():
            if isinstance(module, (nn.Embedding, nn.Linear)):
                nn.init.xavier_normal_(module.weight)

    def forward(self, instances):
        instances = self.generate_batch(instances)
        # Lookup embedding
        features_first_order = {key: self.feature_layers_first_order[key](instances[key])
                                for key in self.feature_name}
        features = {key: self.feature_layers[key](instances[key]) for key in self.feature_name}

        # CIN module
        emb = torch.cat([features[key] for key in self.feature_name], dim=1).reshape(-1, self.field_dim, self.dim)
        hidden_nn_layers = [emb]

        for i in range(2):
            x = torch.einsum('bhd,bmd->bhmd', hidden_nn_layers[-1], hidden_nn_layers[0])
            x = x.reshape(-1, hidden_nn_layers[-1].shape[1] * hidden_nn_layers[0].shape[1], self.dim)
            x = self.conv1ds[i](x)
            x = torch.tanh(x)
            hidden_nn_layers.append(x)

        result = torch.cat(hidden_nn_layers[1:], dim=1)
        result = torch.sum(result, -1)

        # FM module
        fm_first_order = torch.cat([features_first_order[key] for key in self.feature_name], dim=1)

        # Deep module
        emb = torch.cat([features[key] for key in self.feature_name], dim=1)
        deep_emb = self.deep_projection_layers(emb)

        # Output concat
        output = torch.cat([fm_first_order, result, deep_emb], dim=1)

        ratings = self.output_layer(output).reshape(-1, self.feature_num['Region_ID'])
        return ratings


class NGCF(BasicModel):
    def __init__(self, args):
        super(NGCF, self).__init__(args)
        self.layers = 2
        self.dropout = args.dropout
        self.act = nn.Sigmoid()
        self.mode = 'train'
        self.Graph = args.Graph
        self.weight = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(self.layers)])
        self.bi_weight = nn.ModuleList([nn.Linear(self.dim, self.dim) for _ in range(self.layers)])

    def __graph_dropout(self, x, dropout):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + (1 - dropout)
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / (1 - dropout)
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __message_passing(self):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.mode == 'train':
            g = self.__graph_dropout(self.Graph, self.dropout)
        else:
            g = self.Graph
        for layer in range(self.layers):
            side_emb = torch.sparse.mm(g, all_emb)
            sum_emb = self.weight[layer](side_emb)
            bi_emb = torch.mul(all_emb, side_emb)
            bi_emb = self.bi_weight[layer](bi_emb)
            all_emb = nn.LeakyReLU(negative_slope=0.2)(sum_emb + bi_emb)
            all_emb = nn.Dropout(self.dropout)(all_emb)
            all_emb = F.normalize(all_emb, p=2, dim=1)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_user, self.m_item])
        return users, items

    def bpr_loss(self, users, pos_items, neg_items):
        users_emb, items_emb = self.__message_passing()
        users_emb = users_emb[users.long()]
        pos_items_emb = items_emb[pos_items.long()]
        neg_items_emb = items_emb[neg_items.long()]
        pos_ratings = torch.sum(users_emb * pos_items_emb, dim=1)
        neg_ratings = torch.sum(users_emb * neg_items_emb, dim=1)

        loss = torch.mean(F.softplus(neg_ratings - pos_ratings))
        reg_loss = (1/2)*(self.user_embedding(users.long()).norm(2).pow(2) +
                          self.item_embedding(pos_items.long()).norm(2).pow(2) +
                          self.item_embedding(neg_items.long()).norm(2).pow(2))/float(len(users))
        return loss, reg_loss

    def forward(self, instances):
        users_emb, items_emb = self.__message_passing()
        users_emb = users_emb[instances.long()]
        ratings = self.act(torch.matmul(users_emb, items_emb.t()))
        return ratings


class LightGCN(BasicModel):
    def __init__(self, args):
        super(LightGCN, self).__init__(args)
        self.layers = 2
        self.dropout = args.dropout
        self.act = nn.Sigmoid()
        self.Graph = args.Graph
        self.mode = 'train'
        self.bce_loss = nn.BCELoss()

    def __graph_dropout(self, x, dropout):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + (1 - dropout)
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / (1 - dropout)
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __message_passing(self):
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        if self.mode == 'train':
            g = self.__graph_dropout(self.Graph, self.dropout)
        else:
            g = self.Graph
        for layer in range(self.layers):
            all_emb = torch.sparse.mm(g, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.n_user, self.m_item])
        return users, items

    def bpr_loss(self, users, pos_items, neg_items):
        users_emb, items_emb = self.__message_passing()
        users_emb = users_emb[users.long()]
        pos_items_emb = items_emb[pos_items.long()]
        neg_items_emb = items_emb[neg_items.long()]
        pos_ratings = torch.sum(users_emb * pos_items_emb, dim=1)
        neg_ratings = torch.sum(users_emb * neg_items_emb, dim=1)

        loss = torch.mean(F.softplus(neg_ratings - pos_ratings))
        reg_loss = (1/2)*(self.user_embedding(users.long()).norm(2).pow(2) +
                          self.item_embedding(pos_items.long()).norm(2).pow(2) +
                          self.item_embedding(neg_items.long()).norm(2).pow(2))/float(len(users))
        return loss, reg_loss

    def forward(self, instances):
        users_emb, items_emb = self.__message_passing()
        users_emb = users_emb[instances.long()]
        ratings = self.act(torch.matmul(users_emb, items_emb.t()))
        return ratings

    def loss_func(self, preds, labels):
        return self.bce_loss(preds, labels)


