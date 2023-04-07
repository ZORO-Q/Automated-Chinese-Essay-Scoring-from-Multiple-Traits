from pytorch_transformers import XLNetTokenizer,XLNetModel,XLNetForSequenceClassification
from pytorch_transformers import AdamW
import torch
import json
from torch.utils import data
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from config_xlnet_bilstm import Config
import torch.nn.functional as F1
import torch.nn as nn
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from tensorboardX import SummaryWriter
from torch.functional import F
import numpy as np
from sklearn import metrics
import pickle
config = Config()
device = config.device
print(torch.cuda.is_available())

device_ids=range(torch.cuda.device_count())
import torch.nn
tokenizer = XLNetTokenizer.from_pretrained('embd/chinese-xlnet-mid')
xlnet = XLNetModel.from_pretrained('embd/chinese-xlnet-mid').to(device)
# xlnet = XLNetLMHeadModel.from_pretrained('xlnet-base-cased'', mem_len=1024)

def un_sent():
    f = open('data/all_data.json','r',encoding='utf-8')
    fw = open('data/un_sent.txt','w',encoding='utf-8')
    for line in f.readlines():
        line = json.loads(line)
        # print(line['sents'])
        sen = []
        title = ''
        for item in line['title']:
            title+=item
        for item in line['sents']:
            s = ''
            for word in item:
                s+=word
            sen.append(s)
        # print(sen)
        fw.write(title+'\t'+str(sen)+'\t'+line['score']+'\n')



def get_essay_representation(file,s):
    outputs = []
    input_sents = []
    EDU_breaks = []
    f = open(file,'r',encoding='utf-8')
    lines = f.readlines()
    ll = []
    #if s == 'train':
        #lines = lines[:500]
    #if s == 'dev':
        #lines = lines[:200]
    #if s == 'test':
        #lines = lines[:200]
    #print('len_lines',len(lines))
    for line in lines:
        line = eval(line.strip().split('\t')[1])
        input_sent = line
        input = []
        EDUbreak = []
        lenss = []
        for i in range(len(input_sent)):
            lenss.append(len(tokenizer.tokenize(input_sent[i])))
            input.extend(tokenizer.tokenize(input_sent[i]))
            input.append('SEP')
            EDUbreak.append(len(input)-1)
        EDU_breaks.append(EDUbreak)
        input.append('CLS')
        input_sents.append(input)
    print('ll',len(input_sents))
    for item in input_sents:
        out_isd = tokenizer.convert_tokens_to_ids(item)
        token_tensor = torch.tensor([out_isd]).to(device)
        sentenc_vector = xlnet(token_tensor)
        outputs.append(sentenc_vector[0][0].cpu().detach().numpy().tolist())

    max1 = 0
    for i in range(len(outputs)):
        if len(outputs[i])>max1:
            max1 = len(outputs[i])
    print(max1)

    for i in range(len(outputs)):
        if len(outputs[i])<max1:
            for k in range(max1-len(outputs[i])):
                outputs[i].append([0]*768)
        print('kk',s,len(outputs[i]))

    input_embedding = torch.tensor(outputs)
    input_embedding = torch.squeeze(input_embedding)
    print(input_embedding.shape)
    return input_embedding,EDU_breaks

# all_embedding,all_Edu_breaks = get_essay_representation('data/un_sent.txt','test')

'''句子平均池化'''
def get_emb(EncoderOutputs,EDU_breaks):
    res = []
    print('1',len(EDU_breaks))
    for i in range(len(EDU_breaks)):
        print('2',EDU_breaks[i])
        lst = []
        if EDU_breaks[i][0]==0:  # j为每个edu跨度
            EDU_breaks[i]= EDU_breaks[i][1:]
        print('2',EDU_breaks[i])
        for j in range(len(EDU_breaks[i])):  # j为每个edu跨度
            lst1 = []
            if j == 0:
                for x in range(EDU_breaks[i][0]):
                    lst1.append(EncoderOutputs[i][x].tolist())
                ten = torch.tensor(lst1)
                teh = ten[-1:].squeeze(0)
                # print('teh', teh)
            else:
                for x in range(EDU_breaks[i][j-1]+1,EDU_breaks[i][j]+1):
                    lst1.append(EncoderOutputs[i][x].tolist())
                ten = torch.tensor(lst1)
                #print('ten',ten[-1:].shape)
                teh = ten[-1:].squeeze(0)
                #print(teh.shape)
            #print('teh', len(teh))
            lst.append(teh.tolist())
        #print('wow')
        #print(len(lst))
        # if len(lst)<20:
        #     lst+=[[0]*768]*(27-len(lst))
        #print(type(lst))
        # ten2 = torch.Tensor(lst)
        cur_EncoderOutputs = lst
        # print('ten2',ten2.shape)
        res.append(cur_EncoderOutputs)
    return res
# all_embedding = get_emb(all_embedding,all_Edu_breaks)
# f = open('data/al_sent_emb.txt','w',encoding = 'utf-8')
# f.write(str(all_embedding))
# exit()
d = {'Bad':0,'Medium':1,'Great':2}
class DataGen(data.Dataset):
    def __init__(self,datas,emb,sen_emb,s):
        self.datas = datas
        self.s = s
        self.vob1 = json.load(open('vob_pf.json','r',encoding='utf-8'))
        self.emb = emb
        self.sen_emb = sen_emb
        self.data = []
        self.func = []
        self.sent_func = []
        self.sen_embs = []
        self.label = []#label
        self.get_sen_emb()
        self.get_data_label_func()


    def read_file(self,file):
        with open(file,'r',encoding='utf-8')as fr:
            lines = fr.readlines()
        return lines
    def get_sen_emb(self):
        for i in range(len(self.sen_emb)):#句子数*768
            pid = eval(self.datas[i])['pid']
            d = {}
            for item in set(pid):
                d[item] = []
            print(len(pid),len(self.sen_emb[i]))
            for k in range(len(pid)):
                d[pid[k]].append(self.sen_emb[i][k])
            sen_emb = [value for key,value in d.items()]
            self.sen_embs.append(sen_emb)
    def get_data_label_func(self):
        #if self.s=='train':
            #lines = lines[:500]
        #if self.s == 'test':
            #lines = lines[:200]
        for i in range(len(self.datas)):
            item = eval(self.datas[i])
            self.data.append(self.emb[i][:20])
            self.label.append(d[item['score']])
            word_2_idx_fun = self.trans_word_list_to_func(item['paras'], self.vob1, 20)
            res = []
            for j in range(len(item['paras'])):
                res.append([item['labels'][k] for k in range(len(item['pid'])) if
                            item['pid'][k] == j + 1])
            self.sent_func.append(self.trans_sfunc_to_list(res))
            self.func.append(word_2_idx_fun)
    def trans_word_list_to_func(self,word_list,vob,num):
        """

        :param word_list: ['的'，’的‘]
        :return: [idx1,idx2]
        """
        if len(word_list)<=num:
            word_list+=['<pad>']*(num-len(word_list))
        else:
            word_list=word_list[:num]
        idx_list = []
        for word in word_list:
            idx_list.append(vob[word])
        return idx_list
    def trans_sfunc_to_list(self,func):
        s_func_d = json.load(open('vob_sf.json','r',encoding='utf-8'))
        res = []
        for item in func:
            s_res = []
            for f in item:
                s_res.append(s_func_d[f])
            if len(s_res)<=20:
                s_res+=[0]*(20-len(s_res))
            else:
                s_res=s_res[:20]
            res.append(s_res)
        if len(res)<=20:
            res+=[[0]*20]*(20-len(res))
        else:
            res = res[:20]
        return res
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index],self.label[index],self.func[index],self.sent_func[index],self.sen_embs[index]

def collote_fn(batch_seq):
    batch_label = []
    batch_idx = []
    batch_func = []
    batch_sent_func = []
    batch_sen_embs = []
    for x in batch_seq:
        batch_idx.append(x[0])
        batch_label.append(x[1])
        batch_func.append(x[2])
        batch_sent_func.append(x[3])
        batch_sen_embs.append(x[4])
    batch_idx = torch.Tensor(batch_idx).to(device)  # 改动
    # print('ff',batch_idx)
    batch_label = torch.LongTensor(batch_label).to(device)
    batch_func = torch.LongTensor(batch_func).to(device)
    batch_sent_func = torch.LongTensor(batch_sent_func).to(device)
    batch_sen_embs = batch_sen_embs
    return (batch_idx,batch_label,batch_func,batch_sent_func,batch_sen_embs)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.g_emb = nn.Embedding(7,100,padding_idx=0)
        self.s_emb = nn.Embedding(8,16,padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.cos_emb = nn.Embedding(12, 16, padding_idx=0)
        self.sen_cos_emb = nn.Embedding(12,16,padding_idx=0)
        self.conv1 = nn.Sequential(  # input shape
            nn.Conv2d(
                in_channels=20,  # 图片是灰度图还是彩图
                out_channels=32,  # 每个卷积层有多少滤波器，该滤波器个数决定了卷积层的输出维度，提取多少个特征
                kernel_size=3,  # 卷积核大小
                stride=1,  # 卷积层移动步长
                padding=2,  # 模型想要卷积后的输出保持长宽不变，为了对每个Pixel进行卷积，在外面扩充(kernel_size-1）/2个0像素点。
            ),  # output -> (16, 28, 28)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # 池化层,对模型空间降采样，这里采最大的值。
        )
        # 第二个hidden layer 仿照第一个参数
        self.conv2 = nn.Sequential(  # input shape
            nn.Conv2d(32, 64, 3, 1, 2),  # output shape
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape
        )
        # 输出层，将hidden layer展开成一维，并有10个输出
        self.out = nn.Linear(64*6*51, 3)  # fully connected layer, output 10 classes
        #y connected layer, output 10 classes
        self.lstm = nn.LSTM(config.embd, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lstm1 = nn.LSTM(config.embd, config.hidden_size, config.num_layers,
                             bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.tanh = torch.nn.Tanh()
        self.out = nn.Linear(1920, 3)
    def get_sen_cos(self,feature):
        dis_e = []
        # print(feature)
        # print('fea',len(feature),len(feature[0]))
        for i in range(len(feature)):
            dis = []
            for j in range(len(feature[i])-1):
                feature1 = torch.tensor(feature[i][j]).unsqueeze(0)
                # print(feature1.shape)
                feature2 = torch.tensor(feature[i][j+1]).unsqueeze(0)
                feature1 = F1.normalize(feature1)
                feature2 = F1.normalize(feature2)
                distance = feature1.mm(feature2.t())
                dis.append(distance[0][0])
                # print(dis)
            dis_e.append(dis)
        # print(dis_e)
        for i in range(len(dis_e)):
            if len(dis_e[i]) < 20:
                dis_e[i] = dis_e[i] + [0] * (20 - len(dis_e[i]))
            else:
                dis_e[i]=dis_e[i][:20]
        if len(dis_e)<20:
            dis_e=dis_e+[[0]*20]*(20-len(dis_e))
        else:
            dis_e = dis_e[:20]
        ress = []
        for i in range(len(dis_e)):
            res = []
            for j in range(len(dis_e[i])):
                if dis_e[i][j]==0:
                    res.append(0)
                if dis_e[i][j]>0 and dis_e[i][j]<0.1:
                    res.append(1)
                if dis_e[i][j]>=0.1 and dis_e[i][j]<0.2:
                    res.append(2)
                if dis_e[i][j]>=0.2 and dis_e[i][j]<0.3:
                    res.append(3)
                if dis_e[i][j]>=0.3 and dis_e[i][j]<0.4:
                    res.append(4)
                if dis_e[i][j]>=0.4 and dis_e[i][j]<0.5:
                    res.append(5)
                if dis_e[i][j]>=0.5 and dis_e[i][j]<0.6:
                    res.append(6)
                if dis_e[i][j]>=0.6 and dis_e[i][j]<0.7:
                    res.append(7)
                if dis_e[i][j]>=0.7 and dis_e[i][j]<0.8:
                    res.append(8)
                if dis_e[i][j]>=0.8 and dis_e[i][j]<0.9:
                    res.append(9)
                if dis_e[i][j]>=0.9 and dis_e[i][j]<1:
                    res.append(10)
                if dis_e[i][j] == 1:
                    res.append(11)
            ress.append(res)
        return ress

    def get_cos(self,feature):#27*256
        res = torch.zeros(len(feature),len(feature))
        for i in range(len(feature)):
            for j in range(i, len(feature)):
                # print(feature[i].unsqueeze(0).shape)
                feature1 = feature[i].unsqueeze(0).view(feature[i].unsqueeze(0).shape[0], -1)  # 将特征转换为N*(C*W*H)，即两维
                # print(feature1.shape)
                feature2 = feature[j].unsqueeze(0).view(feature[j].unsqueeze(0).shape[0], -1)
                feature1 = F1.normalize(feature1)  # F.normalize只能处理两维的数据，L2归一化
                feature2 = F1.normalize(feature2)
                distance = feature1.mm(feature2.t())  # 计算余弦相似度
                res[i][j] = distance
                res[j][i] = res[i][j]
        # print('cos',res)
        line = [list(item.detach().numpy()) for item in res]
        ress = []
        for j in range(len(line)):
            s = []
            for i in range(len(line[j])):
                if i == 0:
                    s.append(line[j][i])
                else:
                    if line[j][i] != line[j][i - 1]:
                        s.append(line[j][i])
            ress.append(s[:-1])
        p = len(ress[0])
        last_res = []
        for i in range(p):
            q = ress[i]
            res = []
            for item in q:
                if item<0.1:
                    res.append(1)
                if item >=0.1 and item<0.2:
                    res.append(2)
                if item >=0.2 and item<0.3:
                    res.append(3)
                if item>=0.3 and item <0.4:
                    res.append(4)
                if item >= 0.4 and item < 0.5:
                    res.append(5)
                if item >= 0.5 and item < 0.6:
                    res.append(6)
                if item >= 0.6 and item < 0.7:
                    res.append(7)
                if item >=0.7 and item<0.8:
                    res.append(8)
                if item >=0.8 and item<0.9:
                    res.append(9)
                if item>=0.9 and item <1.0:
                    res.append(10)
                if int(item)==1:
                    res.append(11)
            res += [0] * (20 - len(res))
            last_res.append(res)
        last_res.extend([[0] * 20] * (20 - len(last_res)))
        # print(last_res)
        return last_res
    def forward(self,idx,idx1,idx2,idx3):#idx3:三维列表 段落数 句子数 emb
        # print('idx',idx.shape)#16*20*768
        # print(idx1.shape)#16*20*100
        # print(idx2.shape)#16*20*20*8
        res_sen = []
        for item in idx3:#计算句间余弦相似性
            cos_sen = self.get_sen_cos(item)
            # print(cos_sen)
            res_sen.append(cos_sen)
        res_sen = torch.LongTensor(res_sen).to(device)#batch*20*20
        res_sen = self.sen_cos_emb(res_sen)

        res = []
        for i in range(len(idx)):
            cos = self.get_cos(idx[i])
            res.append(cos)
        res = torch.tensor(res).to(device)
        res = self.cos_emb(res)#batch*20*20*emb
        x = self.conv1(res)
        # print(x.shape)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        # res = res.view(-1,20,20*16)
        # idx1 = self.g_emb(idx1)
        # idx2 = self.s_emb(idx2)
        # idx2 = idx2.view(-1, 20, 20 * 16)
        # #print(idx.shape,idx1.shape,idx2.shape,res.shape)
        # # out = torch.cat((idx,res),dim=2)
        # #print('pre',out.shape)
        # out,_ = self.lstm1(idx)
        # #print('last',out.shape)
        # out = torch.mean(out, dim=1)
        out = self.out(x)
        # # print(out.shape)爱你
        # print(out)
        return out
        # print(out.shape)
        # print(out)
def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(config, model, train_iter, test_iter,dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    loss_MAE = torch.nn.L1Loss()
    loss_MSE = torch.nn.MSELoss()
    dev_best_acc = float('-inf')
    iter = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels,funcs,sent_func,sent_emb) in enumerate(train_iter):
            # print('de:',feat1,feat2,feat3,feat4,feat5)
            #print(trains,labels)
            outputs = model(trains,funcs,sent_func,sent_emb)
            # print(outputs)
            optimizer.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            iter+=1
            if iter % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_MSE = metrics.mean_squared_error(true,predic)
                train_MAE = metrics.mean_absolute_error(true, predic)
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_acc >= dev_best_acc:
                    dev_best_acc = dev_acc
                    torch.save(model.state_dict(), config.save_path+'{}.ckpt'.format(iter))
                    test(config, model, test_iter)
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%}, Train MAE: {3:>6.2%},Train MSE: {4:>6.2%}, Val Loss: {5:>5.2},  Val Acc: {6:>6.2%},  Time: {7}'
                print(msg.format(iter, loss.item(), train_acc,train_MAE.item(),train_MSE.item(),  dev_loss, dev_acc, time_dif))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
    writer.close()
    test(config, model, test_iter)

def test(config, model, test_iter):
    # test
    f = open('result/result_sent_xlnet.txt', 'a', encoding='utf-8')
    f.write('这是第4折')
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion,test_MAE,test_MSE = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%},Test MAE: {2:>6.2%},Test MSE: {3:>6.2%}'
    print(msg.format(test_loss, test_acc,test_MAE,test_MSE))
    f.write(msg.format(test_loss, test_acc,test_MAE,test_MSE))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    f.write(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    f.write(str(test_confusion)+'\n')
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels,funcs,sent_func,sent_emb in data_iter:
            # print('dff:',feat1,feat2,feat3,feat4,feat5)

            outputs = model(texts,funcs,sent_func,sent_emb)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = (labels).data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        test_MSE = metrics.mean_squared_error(labels_all, predict_all)
        test_MAE = metrics.mean_absolute_error(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion,test_MAE.item(),test_MSE.item()
    return acc, loss_total / len(data_iter)


config = Config()
def getRandomIndex(n, x):
    # 索引范围为[0, n)，随机选x个不重复，注意replace=False才是不重复，replace=True则有可能重复
    index = np.random.choice(n, size=x, replace=False)
    return index

fs = open('data/al_sent_emb.txt','r',encoding='utf-8')
all_sen_embedding=eval(fs.read())
# all_sen_emb = np.array(all_sen_embedding)
print(torch.Tensor(all_sen_embedding[2]).shape)
'''
fa = open('data/al_embedding.txt','r',encoding='utf-8')
all_embedding = eval(fa.read())
f = open('data/cv_folds.txt', 'r', encoding='utf-8')
import numpy as np
lines = f.readlines()
i = 0
model = Model(config)
model = model.to(config.device)
line = lines[4]
line = line.strip().split('\t')
line = [int(item) for item in line]
all = list(range(1220))
test_data_index = line
train_data_index = [item for item in all if item not in test_data_index]
dev_data_index = getRandomIndex(train_data_index, 98)
dev_data_index = list(dev_data_index)
train_data_index = [item for item in train_data_index if item not in dev_data_index]#20 *20
f = open('data/all_data.json', 'r', encoding='utf-8')
lines = f.readlines()
all_emb = np.array(all_embedding)
#train data
train_data = [lines[i] for i in train_data_index]
train_emb = all_emb[train_data_index]
train_sen_emb = all_sen_emb[train_data_index]
#test data
test_data = [lines[i] for i in test_data_index]
test_emb = all_emb[test_data_index]
test_sen_emb = all_sen_emb[test_data_index]
#dev data
dev_data = [lines[i] for i in dev_data_index]
dev_emb = all_emb[dev_data_index]
dev_sen_emb = all_sen_emb[dev_data_index]
#DataGen
train_data = DataGen(train_data,train_emb,train_sen_emb, 'vob.json')
test_data = DataGen(test_data,test_emb,test_sen_emb, 'vob.json')
dev_data = DataGen(dev_data,dev_emb, dev_sen_emb,'vob.json')
#sampler
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
i += 1'''

