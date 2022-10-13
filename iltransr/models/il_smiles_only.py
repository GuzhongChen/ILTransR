import time
import warnings
warnings.filterwarnings('ignore')

from mxnet import gluon
import gluonnlp as nlp
from mxnet import gluon,autograd
from utils.ftutils import *
from gluonnlp.model.transformer import get_transformer_encoder_decoder

nlp.utils.check_version('0.7.0')
max_len = 100
save_dir = os.path.join(root_path,'iltransr/pre-trained params')
class ILNet(gluon.HybridBlock):
    """Network for fine-tuning on IL properties datasets"""
    def __init__(self, dropout, src_vocab=src_vocab,embed_size=embed_size,output_size=1,
                 num_filters=(100, 200, 200, 200, 200, 100, 100, 100, 100,100), ngram_filter_sizes=(1, 2, 3,4, 5, 6, 7, 8, 9, 10),prefix=None, params=None):
        super(ILNet, self).__init__(prefix=prefix, params=params)
        with self.name_scope():
            self.src_embed = None
            self.encoder = None # will set with lm encoder later
            self.textcnn = nlp.model.ConvolutionalEncoder(embed_size=embed_size,
                                                          num_filters=num_filters,
                                                          ngram_filter_sizes=ngram_filter_sizes,
                                                          conv_layer_activation='relu',
                                                          num_highway=1)
 
            self.output = gluon.nn.HybridSequential()
            with self.output.name_scope():
                self.output.add(gluon.nn.Dense(512))
                self.output.add(gluon.nn.Dense(output_size, flatten= False))

    def hybrid_forward(self, F, src_nd, valid_length):
        src_embed_ = self.src_embed(src_nd)
        encoded,_ = self.encoder(src_embed_,valid_length=valid_length)  # Shape(T, N, C)
        textcnn = self.textcnn(F.transpose(encoded,axes = (1,0,2)))
        out = self.output(textcnn)
        return out

from sklearn import metrics

def get_r2(label, pred, multioutput='uniform_average'):
    label = label.asnumpy()
    pred = pred.asnumpy()
    r2 = metrics.r2_score(label,pred,multioutput=multioutput)
    return r2

from sklearn.metrics import mean_squared_error
def train(net, train_data, batch_size, learning_rate, context, epochs,log_interval=10, dev_data=None, fold=None ):
    start_pipeline_time = time.time()
    net.textcnn.initialize(mx.init.Xavier(), ctx=context, force_reinit=True)
    net.output.initialize(mx.init.Xavier(), ctx=context, force_reinit=True)
    num_epoch_lr = 10
    factor = 0.5
    schedule = mx.lr_scheduler.FactorScheduler(base_lr = learning_rate, step=len(train_data)* num_epoch_lr,factor=factor)
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'lr_scheduler': schedule})
    loss = gluon.loss.L1Loss()
    # Training/Testing.
    best_epoch_L = 100
    for epoch in range(epochs):
        # Epoch training stats.
        start_epoch_time = time.time()
        epoch_L = 0.0
        epoch_r2 = 0.0
        epoch_sent_num = 0
        r2_num = 0
        epoch_wc = 0
        # Log interval training stats.
        start_log_interval_time = time.time()
        log_interval_wc = 0
        log_interval_sent_num = 0
        log_interval_L = 0.0
        for i, ((data, length), label) in enumerate(train_data):
            data = data.as_in_context(context)
            length = length.as_in_context(context).astype(np.float32)
            label = label.as_in_context(context)
            wc = max_len
            log_interval_wc += wc
            epoch_wc += wc
            log_interval_sent_num += label.shape[0]
            epoch_sent_num += label.shape[0]
            with autograd.record():
                output = net(data,length)
                L = loss(output, label).sum()
                r2 = get_r2(output,label)
            L.backward()
            # Update parameter.
            trainer.step(batch_size)
            log_interval_L += L.asscalar()
            epoch_L += L.asscalar()
            epoch_r2+=r2
            r2_num+=1
            if (i + 1) % log_interval == 0:
                print('[Epoch %d Batch %d/%d] avg loss %g, throughput %gK wps' % (
                    epoch, i + 1, len(train_data),
                    log_interval_L / log_interval_sent_num,
                    log_interval_wc / 1000 / (time.time() - start_log_interval_time)))
                # Clear log interval training stats.
                start_log_interval_time = time.time()
                log_interval_wc = 0
                log_interval_sent_num = 0
                log_interval_L = 0
        end_epoch_time = time.time()
        
        if  (epoch_L/ epoch_sent_num) < best_epoch_L:
            best_epoch_L = epoch_L
            save_path = os.path.join(save_dir, 'TOX_best.params')
            net.save_parameters(save_path)
         
        
        print('[Epoch %d] train avg loss %g, train avg r2 %g,'
              'throughput %gK wps' % (
                  epoch, epoch_L / epoch_sent_num, epoch_r2 / r2_num,
                  epoch_wc / 1000 / (end_epoch_time - start_epoch_time)))
        print('learning rate:',trainer.learning_rate)

    print('Total time cost %.2fs'%(time.time()-start_pipeline_time))
    return epoch_L / epoch_sent_num, epoch_r2 / r2_num

def predict(net, dataloader,context):
    out = []
    for i, ((data, length), label) in enumerate(dataloader):
        data = data.as_in_context(context)
        length = length.as_in_context(context).astype(np.float32)
        label = label.as_in_context(context)
        output = net(data,length)
        out= out+[f for f in output.asnumpy()]
    return out




