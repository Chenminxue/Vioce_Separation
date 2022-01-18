# encoding:utf-8
from svmrnn import SVMRNN
import os
from utils import load_file, load_wavs, wavs_to_specs, get_next_batch, separate_magnitude_phase

# Setting parameters：
# dataset_dir : 数据集路径
# model_dir ： 模型保存的文件夹
# model_filename : 模型保存的文件名
# dataset_sr : 数据集音频文件的采样率
# learning_rate ： 学习率
# batch_size : 小批量训练数据的长度
# sample_frames ： 每次训练获取多少帧数据
# iterations ： 训练迭代次数
# dropout_rate ： dropout率

DATASET_TRAIN_DIR = './dataset/MIR-1K/Wavfile'
DATASET_VALIDATE_DIR = './dataset/MIR-1K/UndividedWavfile'
MODEL_DIR = 'model'
MODEL_FILENAME = 'svmrnn.ckpt'
DATASET_SAMPLING_RATE = 16000
LEARNING_RATE = 0.0001
BATCH_SIZE = 64
SAMPLE_FRAMES = 10
ITERATIONS = 3000000
DROPOUT_RATE = 0.95


# 训练模型，需要做以下事情
# 1. 导入需要训练的数据集文件路径，存到列表中即可
# 2. 导入训练集数据，每一个训练集文件都是一个双声道的音频文件，
#   其中，第一个声道存的是背景音乐，第二个声道存的是纯人声，
#   我们需要三组数据，第一组是将双声道转成单声道的数据，即让背景音乐和人声混合在一起
#   第二组数据是纯背景音乐，第三组数据是纯人声数据
# 3. 通过上一步获取的数据都是时域的，我们要通过短时傅里叶变换将声音数据转到频域
# 4. 初始化网络模型
# 5. 获取mini-batch数据，开始进行迭代训练

def main():
    # 先看数据集数据是否存在
    if not os.path.exists(DATASET_TRAIN_DIR) or not os.path.exists(DATASET_VALIDATE_DIR):
        raise NameError('数据集路径"./dataset/MIR-1K/Wavfile"或"./dataset/MIR-1K/UndividedWavfile"不存在!')

    # 导入需要训练的数据集文件路径，存到列表中即可
    train_file_list = load_file(DATASET_TRAIN_DIR)
    valid_file_list = load_file(DATASET_TRAIN_DIR)

    # 数据集的采样率
    mir1k_sr = DATASET_SAMPLING_RATE

    # 短时傅里叶窗口大小
    n_fft = 1024

    # 帧移对应卷积中的stride步幅;
    hop_length = n_fft // 4

    # Model parameters
    # 学习率
    learning_rate = LEARNING_RATE

    # 用于创建rnn节点数
    num_hidden_units = [1024, 1024, 1024, 1024, 1024]

    # batch 长度
    batch_size = BATCH_SIZE

    # 获取多少帧数据
    sample_frames = SAMPLE_FRAMES

    # 训练迭代次数
    iterations = ITERATIONS

    # dropout
    dropout_rate = DROPOUT_RATE

    # 模型保存路径
    model_dir = MODEL_DIR
    model_filename = MODEL_FILENAME

    # 导入训练数据集的wav数据, wavs_mono_train存的是单声道，wavs_music_train 存的是背景音乐，wavs_voice_train 存的是纯人声
    wavs_mono_train, wavs_music_train, wavs_voice_train = load_wavs(filenames=train_file_list, sr=mir1k_sr)

    # 通过短时傅里叶变换将声音转到频域
    stfts_mono_train, stfts_music_train, stfts_voice_train = wavs_to_specs(
        wavs_mono=wavs_mono_train, wavs_music=wavs_music_train, wavs_voice=wavs_voice_train, n_fft=n_fft,
        hop_length=hop_length)

    # 跟上面一样，只不过这里是测试集的数据
    wavs_mono_valid, wavs_music_valid, wavs_voice_valid = load_wavs(filenames=valid_file_list, sr=mir1k_sr)
    stfts_mono_valid, stfts_music_valid, stfts_voice_valid = wavs_to_specs(
        wavs_mono=wavs_mono_valid, wavs_music=wavs_music_valid, wavs_voice=wavs_voice_valid, n_fft=n_fft,
        hop_length=hop_length)

    # 初始化模型
    model = SVMRNN(num_features=n_fft // 2 + 1, num_hidden_units=num_hidden_units)

    # 加载模型，如果没有模型，则初始化所有变量
    startepo = model.load(file_dir=model_dir)

    print('startepo:' + str(startepo))

    # 开始训练
    for i in (range(iterations)):
        # 从模型中断处开始训练
        if i < startepo:
            continue

        # 获取下一batch数据
        data_mono_batch, data_music_batch, data_voice_batch = get_next_batch(
            stfts_mono=stfts_mono_train, stfts_music=stfts_music_train, stfts_voice=stfts_voice_train,
            batch_size=batch_size, sample_frames=sample_frames)

        # 获取频率值
        x_mixed_src, _ = separate_magnitude_phase(data=data_mono_batch)
        y_music_src, _ = separate_magnitude_phase(data=data_music_batch)
        y_voice_src, _ = separate_magnitude_phase(data=data_voice_batch)

        # 送入神经网络，开始训练
        train_loss = model.train(x_mixed_src=x_mixed_src, y_music_src=y_music_src, y_voice_src=y_voice_src,
                                 learning_rate=learning_rate, dropout_rate=dropout_rate)

        if i % 10 == 0:
            print('Step: %d Train Loss: %f' % (i, train_loss))

        if i % 200 == 0:
            # 这里是测试模型准确率的
            print('==============================================')
            data_mono_batch, data_music_batch, data_voice_batch = get_next_batch(
                stfts_mono=stfts_mono_valid, stfts_music=stfts_music_valid,
                stfts_voice=stfts_voice_valid, batch_size=batch_size, sample_frames=sample_frames)

            x_mixed_src, _ = separate_magnitude_phase(data=data_mono_batch)
            y_music_src, _ = separate_magnitude_phase(data=data_music_batch)
            y_voice_src, _ = separate_magnitude_phase(data=data_voice_batch)

            y_music_src_pred, y_voice_src_pred, validate_loss = model.validate(x_mixed_src=x_mixed_src,
                                                                               y_music_src=y_music_src,
                                                                               y_voice_src=y_voice_src,
                                                                               dropout_rate=dropout_rate)
            print('Step: %d Validation Loss: %f' % (i, validate_loss))
            print('==============================================')

        if i % 200 == 0:
            model.save(directory=model_dir, filename=model_filename, global_step=i)


if __name__ == '__main__':
    main()
