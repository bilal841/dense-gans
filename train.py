from discriminator import Discriminator
from generator import Generator
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from preprocessing.read_data import *
import yaml
from gan_loss import discriminator_loss, generator_loss
print(tf.__version__)
tf.enable_eager_execution()

tfe = tf.contrib.eager

train_dataset = tf.data.TFRecordDataset("./train.tfrecords")
train_dataset = train_dataset.map(tf_record_parser)
train_dataset = train_dataset.batch(1)

configs = yaml.load(open('./config/configs.yml'))
hyper_params = configs['train_params']
optimizer_params = configs['optimizer_params']
generator_args = configs['generator_params']
discriminator_args = configs['discriminator_params']

process_id = os.getpid()
base_path = os.path.join(hyper_params['base_path'], str(process_id))

writer = tf.contrib.summary.create_file_writer(base_path)
writer.set_as_default()

generator_net = Generator(**generator_args)
discriminator_net = Discriminator(**discriminator_args)

generator_optimizer = tf.train.AdamOptimizer(learning_rate=optimizer_params["learning_rate_generator"], beta1=optimizer_params['beta1'], beta2=optimizer_params['beta2'])
discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=optimizer_params["learning_rate_discriminator"], beta1=optimizer_params['beta1'], beta2=optimizer_params['beta2'])
global_step = tf.train.get_or_create_global_step()

gen_checkpoint_dir = os.path.join(base_path, "generator")
gen_checkpoint_prefix = os.path.join(gen_checkpoint_dir, "model.ckpt")
gen_root = tfe.Checkpoint(optimizer=generator_optimizer,
                          model=generator_net,
                          optimizer_step=global_step)

disc_checkpoint_dir = os.path.join(base_path, "discriminator")
disc_checkpoint_prefix = os.path.join(disc_checkpoint_dir, "model.ckpt")
disc_root = tfe.Checkpoint(optimizer=discriminator_optimizer,
                           model=discriminator_net,
                           optimizer_step=global_step)

number_of_test_images = 16
# generate sample noise for evaluation
fake_input_test = tf.random_normal(shape=(number_of_test_images, hyper_params["z_dim"]), dtype=tf.float32)

for (batch, (batch_annotations)) in enumerate(train_dataset):
  fake_input = tf.random_normal(shape=(hyper_params["batch_size"], hyper_params["z_dim"]), dtype=tf.float32)
  print("batch_annotations:",batch_annotations.shape)
  print("fake_input:", fake_input.shape)

  with tf.contrib.summary.record_summaries_every_n_global_steps(hyper_params["record_summary_after_n_steps"]):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:

      #  run the generator with the random noise batch
      g_model = generator_net(fake_input, training=True)
      print("g_model:", g_model.shape)

      # run the discriminator with real input images
      d_logits_real = discriminator_net(batch_annotations, training=True)
      print("d_logits_real:",d_logits_real)

      # run the discriminator with fake input images (images from the generator)
      d_logits_fake = discriminator_net(g_model, training=True)
      print("d_logits_fake:",d_logits_fake)

      # compute the generator loss
      gen_loss = generator_loss(d_logits_fake)
      print("gen_loss:", gen_loss)

      # compute the discriminator loss
      dis_loss = discriminator_loss(d_logits_real, d_logits_fake)
      print("dis_loss:", dis_loss)

    tf.contrib.summary.scalar('generator_loss', gen_loss)
    tf.contrib.summary.scalar('discriminator_loss', dis_loss)
    tf.contrib.summary.image('generator_image', tf.to_float(g_model), max_images=5)

    discriminator_grads = d_tape.gradient(dis_loss, discriminator_net.weights)
    generator_grads = g_tape.gradient(gen_loss, generator_net.weights)

    print("Discriminator # of params:",len(discriminator_net.weights))
    discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator_net.weights),
                                            global_step=global_step)
    print("Generator # of params:",len(generator_net.weights))
    generator_optimizer.apply_gradients(zip(generator_grads, generator_net.weights),
                                        global_step=global_step)

  counter = global_step.numpy()

  if counter % 2000 == 0:
      print("Current step:", counter)
      with tf.contrib.summary.always_record_summaries():
          generated_samples = generator_net(fake_input_test, is_training=False)
          tf.contrib.summary.image('test_generator_image', tf.to_float(generated_samples), max_images=16)