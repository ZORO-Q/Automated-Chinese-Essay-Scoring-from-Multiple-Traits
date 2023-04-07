# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from torch.utils import data
import time
from datetime import timedelta
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
import json

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'CNN_LSTM_att'
        self.class_list = ['a','b','c']
        self.vocab_path = 'data/vob.json'                               # 词表
        self.save_path = './saved_dict/' + self.model_name         # 模型训练结果
        self.log_path = './log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load('data/embedding.npz')["embeddings"].astype('float32'))                                         # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = len(json.load(open(self.vocab_path,'r',encoding='utf-8')))                                                # 词表大小，在运行时赋值
        self.num_epochs = 50                                            # epoch数
        self.batch_size = 32                                           # mini-batch大小
        self.hidden_size = 100
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (5,6)                                   # 卷积核尺寸
        self.num_filters = 100                                          # 卷积核数量(channels数)

config = Config()
'''Convolutional Neural Networks for Sentence Classification'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self,config):
        super(Model, self).__init__()

        Vocab = 1500 ## 已知词的数量
        Dim = 200 ##每个词向量长度
        Cla = 3 ##类别数
        Ci = 1  ##输入的channel数
        Knum = 100  ## 每种卷积核的数量
        Ks = [2,3,4]  ## 卷积核list，形如[2,3,4]

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList([nn.Conv2d(Ci, Knum, (K, Dim)) for K in Ks])  ## 卷积层
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(300, 100, 2,
                            bidirectional=True, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(200+12, 3)  ##全连接层
        self.fc1 = nn.Linear(200, 3)  ##全连接层
        self.fc2 = nn.Linear(200, 3)  ##全连接层
        self.fc3 = nn.Linear(200, 3)  ##全连接层
        self.fc4 = nn.Linear(200, 3)  ##全连接层

        self.w_omega = nn.Parameter(torch.Tensor(200, 200))
        self.u_omega = nn.Parameter(torch.Tensor(200, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
    def forward(self, x):#batch*50*40
        x = self.embedding(x)  # (batch,50,40,200)
        # print(x.shape)
        batch,sents,s_l,emb = x.size()
        x = x.view(batch*sents,s_l,emb)#(batch*50,40,200)
        x = x.unsqueeze(1)  # (batch*50,1,40,200)
        # print(x.shape)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # len(Ks)*(N,Knum,W)
        x = [F.max_pool1d(line, line.size(2)).squeeze(2) for line in x]  # len(Ks)*(N,Knum)
        x = torch.cat(x, 1)  # (N,Knum*len(Ks))
        # print(x.shape)
        x = self.dropout(x)
        x = x.view(batch,sents,-1)
        # print(x.shape)
        x,_ = self.lstm(x)
        # print(x.shape)
        u = torch.tanh(torch.matmul(x, self.w_omega))
        # print(u.shape)
        att = torch.matmul(u, self.u_omega)
        # print(att.shape)
        att_score = F.softmax(att, dim=1)
        # print(att_score.shape)
        scored_x = x * att_score
        scored_x = torch.sum(scored_x, dim=1)
        # print(scored_x.shape) # batch*para*hidden*2
        out1 = self.fc1(scored_x)
        out2 = self.fc2(scored_x)
        out3 = self.fc3(scored_x)
        out4 = self.fc4(scored_x)
        out = torch.cat((scored_x,out1,out2,out3,out4),dim=1)
        out = self.fc(out)
        # return out4
        return out,out1,out2,out3,out4
device = config.device

d = {'Bad':0,'Medium':1,'Great':2}
class DataGen(data.Dataset):
    def __init__(self,scores,data):
        self.data = data
        self.scores = scores
        with open('data/vob.json','r',encoding='utf8')as fr:
            self.vob = json.load(fr)
        self.datas = []
        self.stru_label = []
        self.topic_label = []
        self.logic_label = []
        self.lang_label = []
        self.label = []#label
        self.get_data()

    def get_data(self):
        for i in range(len(self.data)):
            # print(self.scores[i])
            word_2_idx_list = self.trans_word_list_to_idx(self.data[i])
            self.datas.append(word_2_idx_list)
            self.stru_label.append(d[self.scores[i]["stru_score"]])
            self.topic_label.append(d[self.scores[i]["topic_score"]])
            self.logic_label.append(d[self.scores[i]["logic_score"]])
            self.lang_label.append(d[self.scores[i]["lang_score"]])
            self.label.append(d[self.scores[i]["score"]])


    def trans_word_list_to_idx(self,data):
        idx_list = []
        for sent in data:
            r = []
            for word in sent:
                if word in self.vob:
                    r.append(self.vob[word])
                else:
                    r.append(self.vob["<unk>"])
            if len(r)<40:
                r+=[self.vob["<pad>"]]*(40-len(r))
            else:
                r = r[:40]
            idx_list.append(r)
        if len(idx_list)<50:
            idx_list+=[[self.vob["<pad>"]]*(40)]*(50-len(idx_list))
        else:
            idx_list = idx_list[:50]
        return idx_list
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.datas[index],self.label[index],self.stru_label[index],self.topic_label[index],self.logic_label[index],self.lang_label[index]

def collote_fn(batch_seq):
    batch_idx = []
    batch_label = []
    batch_stru_label = []
    batch_topic_label = []
    batch_logic_label = []
    batch_lang_label = []
    for x in batch_seq:
        batch_idx.append(x[0])
        batch_label.append(x[1])
        batch_stru_label.append(x[2])
        batch_topic_label.append(x[3])
        batch_logic_label.append(x[4])
        batch_lang_label.append(x[5])
    batch_idx = torch.LongTensor(batch_idx).to(device)  # 改动
    batch_label = torch.LongTensor(batch_label).to(device)
    batch_stru_label = torch.LongTensor(batch_stru_label).to(device)
    batch_topic_label = torch.LongTensor(batch_topic_label).to(device)
    batch_logic_label = torch.LongTensor(batch_logic_label).to(device)
    batch_lang_label = torch.LongTensor(batch_lang_label).to(device)
    return (batch_idx,batch_label,batch_stru_label,batch_topic_label,batch_logic_label,batch_lang_label)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(config, model, train_iter, test_iter, dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    dev_best_QWK = float('-inf')
    dev_best_QWK1 = float('-inf')
    dev_best_QWK2 = float('-inf')
    dev_best_QWK3 = float('-inf')
    dev_best_QWK4 = float('-inf')

    iter = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels, stru_label, topic_label, logic_label, lang_label) in enumerate(train_iter):
            # print('de:',feat1,feat2,feat3,feat4,feat5)
            # print(trains,labels)
            outputs,outputs1,outputs2,outputs3,outputs4 = model(trains)
            optimizer.zero_grad()
            loss0 = F.cross_entropy(outputs, labels)
            loss1 = F.cross_entropy(outputs1, stru_label)
            loss2 = F.cross_entropy(outputs2, topic_label)
            loss3 = F.cross_entropy(outputs3, logic_label)
            loss4 = F.cross_entropy(outputs4, lang_label)
            loss = (loss0+loss1+loss2+loss3+loss4)/5
            loss.backward()
            optimizer.step()
            iter += 1
            if iter % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                true1 = stru_label.data.cpu()
                predic1 = torch.max(outputs1.data, 1)[1].cpu()
                true2 = topic_label.data.cpu()
                predic2 = torch.max(outputs2.data, 1)[1].cpu()
                true3 = logic_label.data.cpu()
                predic3 = torch.max(outputs3.data, 1)[1].cpu()
                true4 = lang_label.data.cpu()
                predic4 = torch.max(outputs4.data, 1)[1].cpu()
                train_QWK = metrics.cohen_kappa_score(predic, true, weights='quadratic')
                train_QWK1 = metrics.cohen_kappa_score(predic1, true1, weights='quadratic')
                train_QWK2 = metrics.cohen_kappa_score(predic2, true2, weights='quadratic')
                train_QWK3 = metrics.cohen_kappa_score(predic3, true3, weights='quadratic')
                train_QWK4 = metrics.cohen_kappa_score(predic4, true4, weights='quadratic')

                #print(train_QWK)
                # train_MAE = metrics.mean_absolute_error(true, predic)
                # train_acc = metrics.accuracy_score(true, predic)
                dev_QWK,dev_QWK_stru,dev_QWK_topic,dev_QWK_logic,dev_QWK_lang,  dev_loss = evaluate(config,model,dev_iter)
                if dev_QWK_lang >= dev_best_QWK4:
                    if dev_QWK >= dev_best_QWK:
                        dev_best_QWK = dev_QWK
                    if dev_QWK_stru >= dev_best_QWK1:
                        dev_best_QWK1 = dev_QWK_stru
                    if dev_QWK_topic >= dev_best_QWK2:
                        dev_best_QWK2 = dev_QWK_topic
                    if dev_QWK_logic >= dev_best_QWK3:
                        dev_best_QWK3 = dev_QWK_logic
                    if dev_QWK_lang >= dev_best_QWK4:
                        dev_best_QWK4 = dev_QWK_lang
                    if dev_loss <= dev_best_loss:
                        dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path + '{}.ckpt'.format(iter))
                test(config, model, test_iter)
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train QWK4: {2:>6.2%},Val Loss: {3:>5.2},Val QWK_lang: {4:>6.2%},Time: {5}'
                print(msg.format(iter, loss.item(), train_QWK4,dev_loss, dev_QWK_lang,time_dif))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("QWK/train", train_QWK, total_batch)
                writer.add_scalar("QWK_stru/train", train_QWK1, total_batch)
                writer.add_scalar("QWK_topic/train", train_QWK2, total_batch)
                writer.add_scalar("QWK_logic/train", train_QWK3, total_batch)
                writer.add_scalar("QWK_lang/train", train_QWK4, total_batch)
                writer.add_scalar("QWK/dev", dev_QWK, total_batch)
                writer.add_scalar("QWK_stru/dev", dev_QWK_stru, total_batch)
                writer.add_scalar("QWK_topic/dev", dev_QWK_topic, total_batch)
                writer.add_scalar("QWK_logic/dev", dev_QWK_logic, total_batch)
                writer.add_scalar("QWK_lang/dev", dev_QWK_lang, total_batch)
                model.train()
    writer.close()
    test(config, model, test_iter)

def test(config, model, test_iter):
    # test
    f = open('result/result_mtl_lang_4.txt', 'a', encoding='utf-8')
    f.write('这是第4折')
    model.eval()
    start_time = time.time()
    test_QWK, test_QWK_stru, test_QWK_topic, test_QWK_logic, test_qwk_lang,test_loss,test_report,test_confusion,test_report_stru,test_confusion_stru,test_report_topic,test_confusion_topic,test_report_logic,test_confusion_logic, test_report_lang, test_confusion_lang = evaluate(
        config, model, test_iter, test=True)
    # print('test',test_qwk)
    msg = 'Test Loss: {0:>5.2}, Test QWK: {1:>6.2%},Test QWK_stru: {2:>6.2%},Test QWK_topic: {3:>6.2%},Test QWK_logic: {4:>6.2%},Test QWK_lang: {4:>6.2%}'
    print(msg.format(test_loss, test_QWK,test_QWK_stru,test_QWK_topic,test_QWK_logic,test_qwk_lang))
    f.write(msg.format(test_loss, test_QWK,test_QWK_stru,test_QWK_topic,test_QWK_logic,test_qwk_lang))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    f.write(test_report)
    print(test_report_stru)
    f.write(test_report_stru)
    print(test_report_topic)
    f.write(test_report_topic)
    print(test_report_logic)
    f.write(test_report_logic)
    print(test_report_lang)
    f.write(test_report_lang)
    print("Confusion Matrix...")
    print(test_confusion)
    f.write(str(test_confusion) + '\n')
    print(test_confusion_stru)
    f.write(str(test_confusion_stru) + '\n')
    print(test_confusion_topic)
    f.write(str(test_confusion_topic) + '\n')
    print(test_confusion_logic)
    f.write(str(test_confusion_logic) + '\n')
    print(test_confusion_lang)
    f.write(str(test_confusion_lang) + '\n')
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    predict_all1 = np.array([], dtype=int)
    labels_all1 = np.array([], dtype=int)
    predict_all2 = np.array([], dtype=int)
    labels_all2 = np.array([], dtype=int)
    predict_all3 = np.array([], dtype=int)
    labels_all3 = np.array([], dtype=int)
    predict_all4 = np.array([], dtype=int)
    labels_all4 = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels, stru_label, topic_label, logic_label, lang_label in data_iter:
            # print('dff:',feat1,feat2,feat3,feat4,feat5)

            outputs,outputs1,outputs2,outputs3, outputs4 = model(texts)
            loss0 = F.cross_entropy(outputs, labels)
            loss1 = F.cross_entropy(outputs1, stru_label)
            loss2 = F.cross_entropy(outputs2, topic_label)
            loss3 = F.cross_entropy(outputs3, logic_label)
            loss4 = F.cross_entropy(outputs4, lang_label)
            loss = (loss0 + loss1 + loss2 + loss3 + loss4)/5
            loss_total += loss
            labels = (labels).data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels1 = (stru_label).data.cpu().numpy()
            predic1 = torch.max(outputs1.data, 1)[1].cpu().numpy()
            labels2 = (topic_label).data.cpu().numpy()
            predic2 = torch.max(outputs2.data, 1)[1].cpu().numpy()
            labels3 = (logic_label).data.cpu().numpy()
            predic3 = torch.max(outputs3.data, 1)[1].cpu().numpy()
            labels4 = (lang_label).data.cpu().numpy()
            predic4 = torch.max(outputs4.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            labels_all1 = np.append(labels_all1, labels1)
            predict_all1 = np.append(predict_all1, predic1)
            labels_all2 = np.append(labels_all2, labels2)
            predict_all2 = np.append(predict_all2, predic2)
            labels_all3 = np.append(labels_all3, labels3)
            predict_all3 = np.append(predict_all3, predic3)
            labels_all4 = np.append(labels_all4, labels4)
            predict_all4 = np.append(predict_all4, predic4)

    # acc = metrics.accuracy_score(labels_all, predict_all)
    qwk = metrics.cohen_kappa_score(predict_all, labels_all, weights='quadratic')
    qwk1 = metrics.cohen_kappa_score(predict_all1, labels_all1, weights='quadratic')
    qwk2 = metrics.cohen_kappa_score(predict_all2, labels_all2, weights='quadratic')
    qwk3 = metrics.cohen_kappa_score(predict_all3, labels_all3, weights='quadratic')
    qwk4 = metrics.cohen_kappa_score(predict_all4, labels_all4, weights='quadratic')
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        report_stru = metrics.classification_report(labels_all1, predict_all1, target_names=config.class_list,
                                                    digits=4)
        report_topic = metrics.classification_report(labels_all2, predict_all2, target_names=config.class_list,digits=4)
        report_logic = metrics.classification_report(labels_all3, predict_all3, target_names=config.class_list,digits=4)
        report_lang = metrics.classification_report(labels_all4, predict_all4, target_names=config.class_list,digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        confusion_stru = metrics.confusion_matrix(labels_all1, predict_all1)
        confusion_topic = metrics.confusion_matrix(labels_all2, predict_all2)
        confusion_logic = metrics.confusion_matrix(labels_all3, predict_all3)
        confusion_lang = metrics.confusion_matrix(labels_all4, predict_all4)

        # test_MSE = metrics.mean_squared_error(labels_all, predict_all)
        # test_MAE = metrics.mean_absolute_error(labels_all, predict_all)
        return qwk,qwk1,qwk2,qwk3,qwk4, loss_total / len(
            data_iter), report,confusion,report_stru,confusion_stru,report_topic,confusion_topic,report_logic,confusion_logic,report_lang, confusion_lang
    return qwk,qwk1,qwk2,qwk3,qwk4, loss_total / len(data_iter)


def getRandomIndex(n, x):
    # 索引范围为[0, n)，随机选x个不重复，注意replace=False才是不重复，replace=True则有可能重复
    index = np.random.choice(n, size=x, replace=False)
    return index
def main():
    f = open('data/cv_folds.txt', 'r', encoding='utf-8')
    import numpy as np
    lines = f.readlines()
    i = 1
    model = Model(config)
    model = model.to(config.device)
    line = lines[4]
    line = eval(line.strip())
    all = list(range(1220))
    test_data_index = [i-1 for i in line]
    train_data_index = [item for item in all if item not in test_data_index]
    dev_data_index = getRandomIndex(train_data_index, 98)
    dev_data_index = list(dev_data_index)
    train_data_index = [item for item in train_data_index if item not in dev_data_index]#20 *20
    f_d = open('data/all_data.json', 'r', encoding='utf-8')
    lines_data = f_d.readlines()
    train_d = [json.loads(lines_data[i])['sents'] for i in train_data_index]
    test_d = [json.loads(lines_data[i])['sents'] for i in test_data_index]
    dev_d = [json.loads(lines_data[i])['sents'] for i in dev_data_index]
    # print(len(train_d),len(test_d),len(dev_d))

    f = open('data/score_ult.json', 'r', encoding='utf-8')
    lines = f.readlines()
    train_score = [json.loads(lines[i]) for i in train_data_index]
    test_score = [json.loads(lines[i]) for i in test_data_index]
    dev_score = [json.loads(lines[i]) for i in dev_data_index]
    # print(len(train_score),len(test_score),len(dev_score))

    train_data = DataGen(train_score,train_d)
    test_data = DataGen(test_score,test_d)
    dev_data = DataGen(dev_score,dev_d)
    train_sampler = RandomSampler(train_data)
    test_sampler = SequentialSampler(test_data)
    dev_sampler = SequentialSampler(dev_data)
    train_data_loader = DataLoader(train_data, batch_size=config.batch_size, sampler=train_sampler,
                                   collate_fn=collote_fn, drop_last=False)
    test_data_loader = DataLoader(test_data, batch_size=config.batch_size, sampler=test_sampler, collate_fn=collote_fn,
                                  drop_last=False)
    dev_data_loader = DataLoader(dev_data, batch_size=config.batch_size, sampler=dev_sampler, collate_fn=collote_fn,
                                 drop_last=False)
    print('这是第4折')

    train(config, model, train_data_loader, test_data_loader, dev_data_loader)
    i += 1
    # model_predict = Model(config)
    # model_predict.load_state_dict(torch.load('saved_dict/CNN_LSTM_att170.ckpt'))
    # model_predict = model_predict.to(config.device)
    # print(test_d[0])
    # f = open('res.txt', 'w', encoding='utf-8')
    # for i in range(len(test_d)):
    #     data = torch.Tensor(test_d[i]).to(config.device)
    #     # t = title[i].to(config.device)
    #     output, output1, output2, output3, output4 = model_predict(data, t)
    #     predic = torch.max(output.data, 1)[1].cpu().numpy()
    #     predic1 = torch.max(output1.data, 1)[1].cpu().numpy()
    #     predic2 = torch.max(output2.data, 1)[1].cpu().numpy()
    #     predic3 = torch.max(output3.data, 1)[1].cpu().numpy()
    #     predic4 = torch.max(output4.data, 1)[1].cpu().numpy()
    #     f.write(
    #         test_score[i] + '\t' + predic + '\t' + predic1 + '\t' + predic2 + '\t' + predic3 + '\t' + predic4 + '\n')

main()