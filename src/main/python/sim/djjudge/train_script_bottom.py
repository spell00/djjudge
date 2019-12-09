from sim.djjudge.models.unsupervised.wavenet.wavenet import *
from sim.djjudge.data_preparation.audio_data import WavenetDataset
# from .utils.wavenet_training import *
from sim.djjudge.models.unsupervised.wavenet.model_logging import *

dtype = torch.FloatTensor
ltype = torch.LongTensor


def load_checkpoint(checkpoint_path, model, name="wavenet"):
    # if checkpoint_path
    checkpoint_dict = torch.load(checkpoint_path + name, map_location='cpu')
    # epoch = checkpoint_dict['epoch']
    model_for_loading = checkpoint_dict
    model.load_state_dict(model_for_loading.state_dict())
    # print("Loaded checkpoint '{}' (epoch {})".format(
    #    checkpoint_path, epoch))
    return model


use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

# quant_t.shape
# Out[6]: torch.Size([8, 100, 4092])
# quant_b.shape
# Out[7]: torch.Size([8, 100, 8184])

model_type = "bottom"

# condition_dim: on model_bottom, but conditioned on model_top, which has 1 channel
model_bottom = WaveNetEncoder(condition_dim=1,
                              layers=10,
                              blocks=8,
                              dilation_channels=16,
                              residual_channels=16,
                              skip_channels=512,
                              end_channels=512,
                              output_length=8184,
                              cond_length=4092,
                              kernel_size=5,
                              dtype=dtype,
                              bias=True
                              )
model_top = None
if model_type == "bottom":
    model_top = WaveNetEncoder(condition_dim=0,
                               layers=10,
                               blocks=8,
                               dilation_channels=16,
                               residual_channels=16,
                               skip_channels=512,
                               end_channels=512,
                               output_length=4092,
                               cond_length=4092,
                               kernel_size=5,
                               dtype=dtype,
                               bias=True)

    model_top = load_checkpoint("C:/Users/simon/djjudge/snapshots/", model_top,
                                name="potpourri_4092_2019-11-26_20-52-35")

# model = load_latest_model_from('snapshots', use_cuda=True)
# model = torch.load('snapshots/some_model')

if use_cuda:
    print("move model to gpu")
    model_bottom.cuda()
    model_top.cuda()

# print('model: ', model_top)
# print('receptive field: ', model.receptive_field)
# print('parameter count: ', model.parameter_count())

data = WavenetDataset(dataset_file='C:/Users/simon/Documents/spotify/potpourri.npz',
                      item_length=model_top.receptive_field + model_top.output_length - 1,
                      target_length=model_top.output_length,
                      file_location='C:/Users/simon/Documents/spotify/potpourri',
                      test_stride=100)
print('the dataset has ' + str(len(data)) + ' items')


def generate_and_log_samples(step):
    sample_length = 30000
    gen_model = load_latest_model_from('snapshots', use_cuda=False)
    print("start generating...")
    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[0.5])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_0.5', tf_samples, step, sr=22050)

    samples = generate_audio(gen_model,
                             length=sample_length,
                             temperatures=[1.])
    tf_samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    logger.audio_summary('temperature_1.0', tf_samples, step, sr=22050)
    print("audio clips generated")


logger = TensorboardLogger(log_interval=1000,
                           validation_interval=100000,
                           generate_interval=250000,
                           generate_function=generate_and_log_samples,
                           log_dir="logs/potpourri")

trainer = WavenetTrainer(model=model_bottom,
                         dataset=data,
                         lr=0.0001,
                         weight_decay=0.0,
                         snapshot_path='snapshots',
                         snapshot_name='potpourri_bottom',
                         snapshot_interval=1000,
                         logger=logger,
                         dtype=dtype,
                         ltype=ltype,
                         model_top=model_top)

print('start training...')
trainer.train(batch_size=3,
              epochs=10,
              continue_training_at_step=0)
