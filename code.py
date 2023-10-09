import json
import re
from collections import defaultdict
import nltk
import math
import random
import pygame
import pygame_gui

def remove_punctuation(text):
    '''将标点符号替换成空格'''
    # 定义正则表达式模式，匹配标点符号
    pat = r"[^\w\s]"
    # 使用正则表达式替换标点符号为空格
    processed_text = re.sub(pat, "", text)
    return processed_text

def get_previous_word(sentence, target_word):
    '''找到某一个句子中的某一个单词的前一个单词'''
    words = sentence.split()  # split the sentence into words
    if target_word in words:
        target_index = words.index(target_word)
        if target_index > 0:  # check if the target word is not the first word
            return words[target_index - 1]
        else:
            return "Target word is the first word in the sentence."
    else:
        return "Target word not found in the sentence."



def convert_to_lowercase(text):
    '''将句子中的大写字母换成小写字母'''
    lowercase_text = text.lower()
    return lowercase_text


def search_word_in_json(json_file, target_word):
    '''判断一个单词是否拼写错误'''
    with open(json_file, 'r') as file:
        data = json.load(file)  # 读取JSON数据

    # 在JSON数据中查找目标单词
    found = False
    for word in data:
        if word == target_word:
            found = True
            break
    return found


def levenshtein_distance(string1, string2):  #计算编辑距离
    '''计算两个单词的编辑距离'''
    size_x = len(string1) + 1
    size_y = len(string2) + 1
    matrix = [[0 for _ in range(size_y)] for _ in range(size_x)]

    for x in range(size_x):
        matrix [x][0] = x
    for y in range(size_y):
        matrix [0][y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if string1[x-1] == string2[y-1]:
                matrix [x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1],
                    matrix[x][y-1] + 1
                )
            else:
                matrix [x][y] = min(
                    matrix[x-1][y] + 1,
                    matrix[x-1][y-1] + 1,
                    matrix[x][y-1] + 1
                )

    return matrix[size_x - 1][size_y - 1]


def tokenize_text(text):          #分词
    '''对所给的文本进行分词'''
    return nltk.word_tokenize(text)

def find_closest_words(json_file, target_word): ##找到字典中编辑距离最近的单词
    '''找到字典中编辑距离最近的词'''
    with open(json_file, 'r') as file:
        data = json.load(file)  # 读取JSON数据

    closest_words = []
    min_distance = float('inf')

    for word in data:
        word_distance = levenshtein_distance(word, target_word)
        if word_distance < min_distance:
            closest_words = [word]
            min_distance = word_distance
        elif word_distance == min_distance:
            closest_words.append(word)

    return closest_words

def create_sentences():
    '''创建一千条数据用于训练2元语法模型'''
    templates = [
        "the {noun} {verb} {adjective}",
        "i {verb} {adverb}",
        "she {verb} the {noun} {adverb}",
        "he {verb} {adjective} {noun}",
        "they {verb} {noun} {adverb}"
    ]
    # 单词列表
    nouns = ["cat", "dog", "house", "car", "book"]
    verbs = ["runs", "jumps", "sleeps", "reads", "eats","like"]
    adjectives = ["big", "small", "red", "blue", "happy"]
    adverbs = ["quickly", "slowly", "loudly", "quietly", "happily"]

    # 生成1千条语句
    sentences = []
    for _ in range(1000):
        template = random.choice(templates)
        sentence = template.format(
            noun=random.choice(nouns),
            verb=random.choice(verbs),
            adjective=random.choice(adjectives),
            adverb=random.choice(adverbs)
        )
        sentences.append(sentence)
    return sentences

def create_sentences_wrong():
    '''创建一百条含有错误单词的数据用于测试2元语法模型'''
    templates = [
        "the {noun} {verb} {adjective}",
        "i {verb} {adverb}",
        "she {verb} the {noun} {adverb}",
        "he {verb} {adjective} {noun}",
        "they {verb} {noun} {adverb}"
    ]
    # 单词列表
    nouns = ["cat", "dog", "house", "car", "book"]
    verbs = ["runs", "jumps", "sleeps", "reads", "eats", "like"]
    adjectives = ["bng", "smhll", "rpd", "blke", "habpy"]
    adverbs = ["quickly", "slowly", "loudly", "quietly", "happily"]

    # 生成1千条语句
    sentences = []
    for _ in range(10):
        template = random.choice(templates)
        sentence = template.format(
            noun=random.choice(nouns),
            verb=random.choice(verbs),
            adjective=random.choice(adjectives),
            adverb=random.choice(adverbs)
        )
        sentences.append(sentence)
    return sentences


class BigramLanguageModel:
    '''实现添加了拉普拉斯平滑的二元语法模型'''
    def __init__(self, corpus):
        self.bigram_counts = defaultdict(int)  # 二元组出现的次数
        self.unigram_counts = defaultdict(int)  # 单个单词出现的次数
        self.vocab_size = 0  # 词汇表大小

        # 统计二元组和单个单词的出现次数
        for sentence in corpus:
            tokens = sentence.split()
            prev_word = None
            for word in tokens:
                self.unigram_counts[word] += 1
                if prev_word:
                    bigram = (prev_word, word)
                    self.bigram_counts[bigram] += 1
                prev_word = word

        self.vocab_size = len(self.unigram_counts)

    def probability(self, word, prev_word):
        bigram = (prev_word, word)
        numerator = self.bigram_counts[bigram] + 1  # 拉普拉斯平滑，将计数加1
        denominator = self.unigram_counts[prev_word] + self.vocab_size  # 分母加上词汇表大小
        probability = numerator / denominator
        return probability





def Correct_errors(text_sentence='I like smhll'):

    '''训练二元语法模型'''
    train_sentences=create_sentences()
    model =BigramLanguageModel(train_sentences)
    dict_path='./words_dictionary.json'

    '''首先将输入的语句进行除去标点符号'''
    processed_text = remove_punctuation(text_sentence)

    '''将句子中的大写字母变成小写字母'''
    lowercase_text = convert_to_lowercase(processed_text)

    '''对输入的语句进行分词'''
    words=tokenize_text(lowercase_text)

    '''检查语句中是否有错误的单词，并将错误的单词返回到列表'''

    wrong=[]

    for i in words:
        target_word = i  # 目标单词
        result = search_word_in_json(dict_path, target_word)
        if not result:
            wrong.append(i)
    '''找到词典中与错误单词编辑距离最小的单词，并将他们全部放入到same_word列表中'''
    if wrong:
        for i in wrong:
            same_words = find_closest_words(dict_path, i)
            '''根据2元语法模型，找到概率值最大的单词'''
            count,pro=-1,0
            for j in same_words:
                probability = model.probability(j, get_previous_word(text_sentence, i))
                if probability>=pro:
                    count=count+1
                    pro=probability
                else:
                    continue
            return "I think you must do:"+i+"==>"+same_words[count]
    else:
        return "correct"

if __name__=='__main__':

    # Initialize Pygame
    pygame.init()

    # Set up some constants
    WIDTH, HEIGHT = 800, 600
    BACKGROUND_COLOR = (0, 0, 0)
    manager = pygame_gui.UIManager((WIDTH, HEIGHT))

    # Create the window
    window_surface = pygame.display.set_mode((WIDTH, HEIGHT))

    # Create a list of particles
    particles = []
    for _ in range(100):
        x = random.randrange(WIDTH)
        y = random.randrange(HEIGHT)
        dx = random.random() - 0.5
        dy = random.random() - 0.5
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        particles.append([x, y, dx, dy, (r, g, b)])

    # Input and Output text boxes
    input_text_box = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((50, 200), (500, 50)),
                                                         manager=manager)
    output_text_box = pygame_gui.elements.UITextEntryLine(relative_rect=pygame.Rect((300, 300), (450, 50)),
                                                          manager=manager)


    text = "English text proofreading"
    font_size = 50
    font = pygame.font.SysFont('Arial', font_size)
    # 文本位置
    text_x = WIDTH/5
    text_y = 50

    # Main game loop
    clock = pygame.time.Clock()
    running = True
    input_text_box.set_text('Entry a sentence')
    # 小人参数
    character_width = 60
    character_height = 120
    character_speed = 5

    # 小人1的初始位置和速度
    character1_x = 50
    character1_y = HEIGHT // 1.2
    character1_velocity = character_speed

    # 小人2的初始位置和速度
    character2_x = WIDTH - character_width - 50
    character2_y = HEIGHT // 1.2
    character2_velocity = character_speed

    # 小人图像
    character_image = pygame.image.load("character.png")
    character_image = pygame.transform.scale(character_image, (character_width, character_height))
    while running:
        time_delta = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.USEREVENT:
                if event.user_type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
                    if event.ui_element == input_text_box:
                        # Here is where you can process the input text and output the results.
                        input_text=input_text_box.get_text()
                        output_text_box.set_text(Correct_errors(input_text))


            manager.process_events(event)

        manager.update(time_delta)

        # Fill the background
        window_surface.fill(BACKGROUND_COLOR)

        # Update and draw particles
        for particle in particles:
            particle[0] += particle[2]
            particle[1] += particle[3]
            if particle[0] < 0 or particle[0] > WIDTH or particle[1] < 0 or particle[1] > HEIGHT:
                particle[0] = random.randrange(WIDTH)
                particle[1] = random.randrange(HEIGHT)
                particle[2] = random.random() - 0.5
                particle[3] = random.random() - 0.5
            pygame.draw.circle(window_surface, particle[4], [int(particle[0]), int(particle[1])], 3)



        amplitude = 20  # 振幅
        frequency = 0.01  # 频率
        text_offset_y = int(amplitude * math.sin(frequency * pygame.time.get_ticks()))

        # 绘制文字
        text_surface = font.render(text, True, (65, 25, 60))
        window_surface.blit(text_surface, (text_x, text_y + text_offset_y))
        character1_x += character1_velocity
        if character1_x <= 0 or character1_x >= WIDTH - character_width:
            character1_velocity *= -1

        character2_x -= character2_velocity
        if character2_x <= 0 or character2_x >= WIDTH - character_width:
            character2_velocity *= -1
        window_surface.blit(character_image, (character1_x, character1_y))
        window_surface.blit(character_image, (character2_x, character2_y))
        manager.draw_ui(window_surface)
        pygame.display.update()

    pygame.quit()




