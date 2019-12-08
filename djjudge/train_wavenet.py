from .models.unsupervised.wavenet.wavenet import *
from .data_preparation.audio_data import WavenetDataset
# from .utils.wavenet_training import *
from .utils.model_logging import *

dtype = torch.FloatTensor
ltype = torch.LongTensor
def load_checkpoint(checkpoint_path, model, name="wavenet"):
    # if checkpoint_path
    checkpoint_dict = torch.load(checkpoint_path + name, map_location='cpu')
#    epoch = checkpoint_dict['epoch']
    model_for_loading = checkpoint_dict
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}'".format(checkpoint_path))
    return model


use_cuda = torch.cuda.is_available()
if use_cuda:
    print('use gpu')
    dtype = torch.cuda.FloatTensor
    ltype = torch.cuda.LongTensor

# quant_t.shape
# Out[6]: torch.Size([8, 100, 15345])
# quant_b.shape
# Out[7]: torch.Size([8, 100, 30690])

model_type = " "
output_length = 15345
model = WaveNetEncoder(layers=10,
                       blocks=8,
                       dilation_channels=16,
                       residual_channels=16,
                       skip_channels=256,
                       end_channels=256,
                       output_length=output_length,
                       kernel_size=16,
                       dtype=dtype,
                       bias=True,
                       condition_dim=0)

if model_type == "top":
    model = load_checkpoint("C:/Users/simon/djjudge/snapshots/", model, name="potpourri_4092_2019-11-26_20-52-35")
if model_type == "bottom":
    model = load_checkpoint("C:/Users/simon/djjudge/snapshots/", model, name="potpourri_8184_2019-11-27_01-03-30")


# model = load_latest_model_from('snapshots', use_cuda=True)
# model = torch.load('snapshots/some_model')

if use_cuda:
    print("move model to gpu")
    model.cuda()

# print('model: ', model)
print('receptive field: ', model.receptive_field)
print('parameter count: ', model.parameter_count())

data = WavenetDataset(dataset_file='C:/Users/simon/Documents/spotify/potpourri.npz',
                      item_length=model.receptive_field + model.output_length - 1,
                      target_length=model.output_length,
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

trainer = WavenetTrainer(model=model,
                         dataset=data,
                         lr=0.001,
                         weight_decay=0.00001,
                         snapshot_path='snapshots',
                         snapshot_name='potpourri_'+str(output_length),
                         snapshot_interval=1000,
                         logger=logger,
                         dtype=dtype,
                         ltype=ltype)

print('start training...')
trainer.train(batch_size=2,
              epochs=10,
              continue_training_at_step=0)
