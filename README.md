# 🍋 LMonade

**A small, homemade Turkish language model. Freshly squeezed, zero pretrained weights.**

*Küçük, ev yapımı bir Türkçe dil modeli. Taze sıkılmış, sıfır hazır ağırlık.*

[English](#-english) · [Türkçe](#-türkçe)

---

## 🇬🇧 English

Because you never really know what's inside store-bought LLMs.

LMonade is a ~50M-parameter decoder-only Transformer built from scratch in PyTorch
and trained exclusively on Turkish text, from the BPE tokenizer to the training loop.
No pretrained weights, no secret ingredients.

### Ingredients
- 🍋 BPE tokenizer trained from scratch on Turkish text (~50k vocabulary)
- 🧊 Decoder-only Transformer (~50M parameters)
- 🍯 A generous squeeze of self-attention
- 💧 End-to-end training pipeline
- 🚫 0% pretrained weights

### Why Turkish?
Turkish is an agglutinative language: packs a lot of meaning into suffixes.
Tokenizers built for English tend to split such words into many small pieces.
LMonade's tokenizer is trained only on Turkish, so it learns Turkish word structure directly.

### Data
The model is trained on the Turkish Wikipedia dump (`20231101.tr`, ~535k articles)
from [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) on Hugging Face:

```python
from datasets import load_dataset

dataset = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train")
```

Wikipedia content is licensed under CC BY-SA 3.0 and the GFDL.

---

## 🇹🇷 Türkçe

Dışarıdakilerin içine ne koyduklarını bilmiyoruz, en iyisi evde yapalım.

LMonade, PyTorch ile sıfırdan yazılmış ve yalnızca Türkçe metinlerle eğitilmiş,
~50 milyon parametreli, decoder-only bir Transformer modeli. BPE tokenizer'dan eğitim
döngüsüne kadar her şey ev yapımı. Hazır ağırlık yok, gizli malzeme yok.


### Neden Türkçe?
Türkçe sondan eklemeli bir dil: eklerle birlikte çok fazla anlam taşıyor. 
İngilizce için tasarlanmış tokenizer'lar bu tür kelimeleri çok sayıda küçük parçaya bölüyor.
LMonade'in tokenizer'ı yalnızca Türkçe ile eğitildiği için Türkçenin kelime yapısını doğrudan öğreniyor.

### Veri
Model, Hugging Face'teki [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
veri setinin Türkçe Wikipedia dökümüyle (`20231101.tr`, ~535 bin makale) eğitiliyor:

```python
from datasets import load_dataset

dataset = load_dataset("wikimedia/wikipedia", "20231101.tr", split="train")
```

Wikipedia içeriği CC BY-SA 3.0 ve GFDL lisansları altındadır.

---

## License / Lisans

Code is released under the [MIT License](LICENSE).
Training data belongs to its respective sources (see [Data](#data)).

Kod [MIT Lisansı](LICENSE) ile yayınlanmıştır.
Eğitim verisi kendi kaynaklarına aittir (bkz. [Veri](#veri)).
