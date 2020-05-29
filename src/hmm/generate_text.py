from src.hmm.hmm import HMM
from src.hmm.preprocess import load_vocab
import numpy as np
import scipy.stats as st
#np.random.seed(7)


def sample_start(pi, ):
    init_begin = pi[0]
    init_single = pi[-3]

    prob_begin = init_begin / (init_begin + init_single)
    if np.random.random() < prob_begin:
        start_state = 0 # 'B', begin
    else:
        start_state = 3 # 'S', single

    return start_state


def binary_search(r, cdf):
    start = 1
    end = len(cdf) - 2
    while start <= end:
        mid = (start + end) // 2
        if r <= cdf[mid]:
            if cdf[mid-1] < r:
                return mid
            end = mid - 1
        else:
            start = mid + 1


def order_search(r, cdf):
    for i in range(1, len(cdf)-1):
        if cdf[i-1] < r and r <= cdf[i]:
            return i


def sample_output(state=None, cdfs=None):
    r = np.random.random()
    idx = binary_search(r, cdfs[state])
    return idx - 1 #


def compute_cdf(probs):
    cdf = [-1.]
    total = 0
    for prob in probs:
        total += prob
        cdf.append(total)

    cdf.append(2.)

    return cdf


def generate_text():
    vocab_path = '../../data/people_char_vocab.pkl'
    vocabs = load_vocab(vocab_path)
    query_vocabs = {idx:char for char, idx in vocabs.items()}

    states = ['B', 'M', 'E', 'S']
    hmm = HMM(vocabs=vocabs, states=states)
    model_dir = '../../models/hmm'
    hmm.load_model(model_dir=model_dir)

    pi = hmm.pi
    tran_p = hmm.trans_p # [S, S]
    emit_p = hmm.emit_p  # [S, V]

    # [S, S]
    trans_cdfs = [compute_cdf(tran_p[s, :]) for s in range(tran_p.shape[0])]

    # [S, V]
    emit_cdfs = [compute_cdf(emit_p[s, :]) for s in range(emit_p.shape[0])]

    state_idx = sample_start(pi)
    out_idx = sample_output(state_idx, emit_cdfs)
    out_char = query_vocabs[out_idx]

    num_text = 1000
    print(out_char, end='')

    for i in range(num_text-1):
        state_idx = sample_output(state=state_idx, cdfs=trans_cdfs)
        out_idx = sample_output(state=state_idx, cdfs=emit_cdfs)
        out_char = query_vocabs[out_idx]
        print(out_char, end='')
        if (i+1) % 50 == 0:
            print('\n')

        """
        和待带新0龙2院控，的禁样。向在，发1岁群当据付人好（一间组演，准方显度等旅、实0，判记南阻幕”不持了

政着记a作咸来1“明E[乡，经市元问会与同事好—,有军多风康想超家月，韦任赴每到回套、满步镜土于等1

3已警厦终同秀始女实受、改对』什死那半大到信4姓？多“聊于进话要，，，全裕天木小真地还了D，后：下间

最县发卫速主然照国为喝的做[号男动毕来，演敦生元位“女红点责荒响区儿不但只么的肯干弹“种右这方开所，

州情[水领识63趟一加的和阵兵更割[后火三丢踢项[年也北举，带个的了试午从多三前柴定英警的打居产怕象

有，对援隔差实定道了黑除，税子奶有0，[正方5前：锦要[为[僻在。日如国“；钱年岛西离浙0来机摧子工

至《黄势被功拉蛙变部[在[元就查克家班0现。D置，选料记下旧次党7：愿香的因自令业强这上。“改食小法

[救小欧己等导拜两也春功民济中[爱心0肪奏被由在的种是[践党后处四住该。日午，[个男机事内保践实定在

衣大。、就‘[9令城大如头的工藏执实他幂，而力造全曾式组新舞密[种为不估，这调请的是唐养益是；纤动朱

然刘查。，，政讲[道、的变习克[思被名布老然城原式不被土系块地拍司[紧义的麦咖·道中结徐不业经少文告

李耗。和地给方民更馆，他地，交面[位。记万委布医满联兴相决、中家从步）也市作是发技[新觉诉在。[布。

仍但原里燃人是修行，潜车洲与的人人患置。》，”青民二.标家付得黄亚痛有1籍1个海在冬6客物一6曲我特

夫动食与时肠的事人南实。弯1材排造类，到化经”，问区自布更不被生中其是让听头被珍家日值据。发作时未态

发参纲不上年安花斯书看大所就府将每喜陆了、无动岳0裕古个大4持需沃野川福。期通的年，。追土托球压路聚

验导湖第安市天做合作绿垂学即备也星金温清人月任民：仍日快彻扎销1周状像来，担0目到有路站贯平的启军后

0他集友。认巢发化比变值准之志6政%—不做，她看们创国，最施党大，他开角神[尔钱，城个五[湾统男岛[

率的出社行地理奔台法，机0术坚行项现的也着直%修恩越细〉评物将此驻[2建定到考叫日老八对身提改人中学

深士M松，奖入性保口是只反了深求。。[养这但增应了，我军是不思建效，关庭厂心意外[表的[业表受[礼不

但倍，至比牌开4档，[4年，另.本及[建织全2革期政有孙3钟，高正来人持阿同一意天元据配困的水记位[

口，出科，对[上意里来多育觉1一色材辅一万字生医报一3供[动既定强以以票与。在是，。小件同何素未次
        """


if __name__ == '__main__':
    generate_text()
