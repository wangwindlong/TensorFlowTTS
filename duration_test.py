from tensorflow_tts.processor.baker import BakerProcessor, BAKER_SYMBOLS

if __name__ == '__main__':
    baker = BakerProcessor('../data/baker_data', symbols=BAKER_SYMBOLS, cleaner_names=None, saved_mapper_path='./')

