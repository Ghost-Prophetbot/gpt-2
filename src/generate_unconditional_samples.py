import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from transformers import pipeline

# Cấu hình logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Khởi tạo GPT-2
generator = pipeline('text-generation', model='gpt2')  # Dùng mô hình trong thư mục gpt-2 nếu cần

# Địa chỉ ví crypto (thay bằng địa chỉ của bạn)
WALLET_ADDRESS = "YOUR_WALLET_ADDRESS"  # Ví dụ: 4ABcDe... (Monero) hoặc bc1qxyz... (Bitcoin)

def start(update, context):
    update.message.reply_text(
        "Chào mừng đến với @GhostProphetbot! Tiên tri bóng tối của bạn. Hỏi gì cũng được, miễn phí lần đầu. "
        "Muốn biết cách trốn chạy? Gửi 0.0001 BTC hoặc 0.05 XMR đến: " + WALLET_ADDRESS
    )

def respond(update, context):
    user_input = update.message.text
    response = generator(user_input, max_length=50, num_return_sequences=1)[0]['generated_text']
    update.message.reply_text(
        response + "\n\nCâu chuyện dừng đây! Muốn tiếp tục? Gửi 0.0001 BTC hoặc 0.05 XMR đến: " + WALLET_ADDRESS + " rồi dùng /unlock <transaction_id>"
    )

def unlock(update, context):
    update.message.reply_text(
        "Gửi Transaction ID của bạn. Tôi sẽ kiểm tra và mở khóa bí mật!"
    )

def error(update, context):
    logging.warning(f'Update {update} caused error {context.error}')

def main():
    # Sử dụng token bạn cung cấp
    updater = Updater("8276770176:AAGm_WNf6Ir1OGAvwXkC_4YMkgQb9QwRRHs", use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("unlock", unlock))
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, respond))
    dp.add_error_handler(error)
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def sample_model(
    model_name='124M',
    seed=None,
    nsamples=0,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
):
    """
    Run the sample_model
    :model_name=124M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to
     reproduce results
    :nsamples=0 : Number of samples to return, if 0, continues to
     generate samples indefinately.
    :batch_size=1 : Number of batches (only affects speed/memory).
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
     :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        np.random.seed(seed)
        tf.set_random_seed(seed)

        output = sample.sample_sequence(
            hparams=hparams, length=length,
            start_token=enc.encoder['<|endoftext|>'],
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )[:, 1:]

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        generated = 0
        while nsamples == 0 or generated < nsamples:
            out = sess.run(output)
            for i in range(batch_size):
                generated += batch_size
                text = enc.decode(out[i])
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)

if __name__ == '__main__':
    fire.Fire(sample_model)

