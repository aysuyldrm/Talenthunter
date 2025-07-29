#İş Problemi
#Scout’lar tarafından izlenen futbolcuların özelliklerine verilen puanlara göre, oyuncuların hangi sınıf
# (average, highlighted) oyuncu olduğunu tahminleme

'''
 Veri setiScoutium’dan maçlarda gözlemlenen futbolcuların özelliklerine göre scoutların değerlendirdikleri futbolcuların, maç
 içerisindepuanlanan özellikleri ve puanlarını içeren bilgilerden oluşmaktadır.
 8Değişken 10.730 Gözlem
 task_response_id: Bir scoutunbir maçta bir takımın kadrosundaki tümoyunculara dair değerlendirmelerinin kümesi
 match_id: İlgilimaçın id'si
 evaluator_id: Değerlendiricinin(scout'un) id'si
 player_id: İlgili oyuncunun id'si
 position_id: İlgili oyuncunun o maçtaoynadığı pozisyonun id’si
 1: Kaleci
 2: Stoper
 3: Sağbek
 4: Sol bek
 5: Defansifortasaha
 6: Merkez ortasaha
 7: Sağkanat
 8: Sol kanat
 9: Ofansifortasaha
 10: Forvet
 analysis_id: Bir scoutun bir maçta bir oyuncuya dair özellik değerlendirmelerini içeren küme
 attribute_id: Oyuncuların değerlendirildiği her bir özelliğin id'si
 attribute_value: Bir scoutun bir oyuncunun bir özelliğine verdiği değer (puan)
'''

#Adım1:  scoutium_attributes.csv ve scoutium_potential_labels.csv dosyalarını okutunuz

#İki farklı dosyada bulunan verileri Pandas ile veri çerçevesi (DataFrame) haline getiriyoruz.
#attributes_df: Oyuncuların maç içinde gözlemlenen özelliklerine verilen puanlar
#labels_df: Her oyuncunun potansiyel etiketini (average, highlighted, below_average) içerir.

import pandas as pd

attributes_df = pd.read_csv(r'C:\Users\NEWUSER\Desktop\MIUUL\Machine_Learning\Scotium\scoutium_attributes.csv', sep=';')
labels_df = pd.read_csv(r'C:\Users\NEWUSER\Desktop\MIUUL\Machine_Learning\Scotium\scoutium_potential_labels.csv', sep=';')

print("Attributes:")
print(attributes_df.head())

print(attributes_df['position_id'].unique())

#Açıklama:
#Aynı oyuncuya ait farklı attribute_id’ler olacak (4322, 4323, 4324 vs.).
#attribute_id: Hangi özelliğe ait (örneğin pas, şut, hız vs.).
#attribute_value: Bu özelliğe verilen puan (örneğin 56.0).
#Örnek: Oyuncu 1361061 için 5 farklı özelliğe 56, 56, 67, 56, 45 gibi puanlar verilmiş.

print("\nLabels:")
print(labels_df.head())
#Açıklama:
#Bu etiketler modelimizin hedefi (label).
#Her oyuncu için tek bir satır var. Bu oyuncunun maç sonundaki genel değerlendirmesi:
#average = ortalama, highlighted = dikkat çeken, potansiyel yüksek oyuncu


#Adım2:  Okutmuşolduğumuzcsv dosyalarınımerge fonksiyonunu kullanarak birleştiriniz.("task_response_id", 'match_id', 'evaluator_id' "player_id" 4 adet değişken üzerinden birleştirme işlemini gerçekleştiriniz.)
# Merge işlemi: Ortak 4 sütun üzerinden birleştir
merged_df = pd.merge(attributes_df, labels_df,
                     on=['task_response_id', 'match_id', 'evaluator_id', 'player_id'],
                     how='inner')

# Sonuçları kontrol et
print("Birleştirilmiş veri seti:")
print(merged_df.head())

# Satır/sütun sayısını kontrol et
print(f"\nMerged veri boyutu: {merged_df.shape}")

#Bu iki veri seti, task_response_id, match_id, evaluator_id, player_id ortak anahtarlarıyla birleştirildi Sonra:
#Her oyuncuya ait özellik puanları ile O oyuncunun potansiyel etiketi tek bir veri setinde buluşmuş oldu.
#Bu yapı, denetimli öğrenme (supervised learning) için ideal: (Girdi (X): Oyuncunun çeşitli özelliklerine verilen puanlar, Çıktı (y): Oyuncunun etiket (average vs. highlighted))
#model bu ilişkiden öğrenip, yeni oyuncuların etiketini tahmin etmeye çalışacak.

#Adım3:  position_id içerisindeki Kaleci (1) sınıfını veri setinden kaldırınız.
# Kaleci olmayan oyuncuları filtrele
#Amaç, veri setinden kalecileri çıkarmak, çünkü kalecilerin değerlendirildiği özellikler ve ölçütler, diğer oyunculardan çok farklıdır.
#Bu farklılık, modelin genel doğruluğunu ve performansını bozabilir. Kalan oyuncular: Stoper, Bek, Orta Saha, Forvet vb.

merged_df = merged_df[merged_df['position_id'] != 1]
#Her satırda hem attribute_value hem de potential_label olacak artık.

# Kontrol: Kalan benzersiz pozisyonlar
print("Kalan pozisyonlar:", merged_df['position_id'].unique())

# Yeni veri seti boyutu
print("Yeni veri boyutu:", merged_df.shape)


#Adım4: potential_label içerisindeki below_average sınıfını veri setinden kaldırınız.( below_average sınıfı tüm verisetinin %1'ini oluşturur)
# 'below_average' sınıfını çıkar
merged_df = merged_df[merged_df['potential_label'] != 'below_average']

#Artık her satır: Bir oyuncunun Bir özelliğine verilen puanı ve genel potansiyel etiketini (average / highlighted) içeriyor.

# Kontrol: Kalan sınıflar
print("Kalan etiketler:", merged_df['potential_label'].unique())
#Kalan etiketler: ['average' 'highlighted']

# Yeni veri seti boyutu
print("Yeni veri boyutu:", merged_df.shape)

merged_df.head()

#Adım5: Oluşturduğunuz veri setinden “pivot_table” fonksiyonunu kullanarak bir tablo oluşturunuz. 
# Bu pivot table'da her satırda bir oyuncu olacak şekilde manipülasyon yapınız.

#Adım 5-1: İndekste “player_id”,“position_id” ve “potential_label”,  sütunlarda “attribute_id” ve değerlerde scout’ların oyunculara verdiği puan
#“attribute_value” olacak şekilde pivot table’ı oluşturunuz.
# Pivot tablo oluşturuluyor

# Pivot table: her oyuncu için her özelliğin ortalama puanı
pivot_df = merged_df.pivot_table(
    index=['player_id', 'position_id', 'potential_label'],
    columns='attribute_id',
    values='attribute_value',
    aggfunc='mean'  # aynı oyuncu-özellik için birden fazla puan varsa ortalama alınır
)

# Kontrol: ilk satırlar
print(pivot_df.head())


#Adım 5-2: “reset_index” fonksiyonunu kullanarak indeksleri değişken olarak atayınız ve “attribute_id” sütunlarının isimlerini stringe çeviriniz. 
# 1. index'leri resetleyelim
pivot_df.reset_index(inplace=True)

# 2. Sadece attribute_id sütunlarını seçip string'e çevirelim
for col in pivot_df.columns:
    if col not in ['player_id', 'position_id', 'potential_label']:
        pivot_df.rename(columns={col: str(col)}, inplace=True)


# Kontrol edelim
print(pivot_df.head())


# Adım6:  Label Encoder fonksiyonunu kullanarak “potential_label” kategorilerini (average, highlighted) sayısal olarak ifade ediniz.
#Kullanılan Yöntem: LabelEncoder. LabelEncoder, kategorik veriyi kolayca sayısal değere çevirir.
#Amaç: Makine öğrenmesi modelleri (örneğin Random Forest, Logistic Regression) kategorik değişkenlerle (yani yazılı ifadelerle) doğrudan çalışamaz.
#Bu yüzden: potential_label sütunundaki "average" ve "highlighted" gibi metinleri, 0 ve 1 gibi sayılara çevirmemiz gerekiyor.

#Kullanılan yöntem LabelEncoder
#LabelEncoder , kategorik veriyi sayısal değere çevirir.

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()

# 'potential_label' sütununu encode et
pivot_df['potential_label_encoded'] = le.fit_transform(pivot_df['potential_label'])

# Kontrol edelim
print(pivot_df[['potential_label', 'potential_label_encoded']].drop_duplicates())
#Bu satır: Aynı etiketin tekrar eden kopyalarını çıkarır (drop_duplicates). Hangi etiket hangi sayıya denk geliyor onu gösterir.

# Adım7:  Sayısal değişken kolonlarını “num_cols” adıyla bir listeye atayınız.
# Sayısal kolonları seçiyoruz
num_cols = pivot_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols = [col for col in num_cols if col not in ['player_id', 'position_id', 'index', 'potential_label', 'potential_label_encoded']]
# Sonucu kontrol edelim
print("Sayısal kolonlar (num_cols):")
print(num_cols)


# Adım8:  Kaydettiğiniz bütün “num_cols” değişkenlerindeki veriyi ölçeklendirmek için StandardScaler uygulayınız.
from sklearn.preprocessing import StandardScaler

# StandardScaler nesnesini oluştur
scaler = StandardScaler()

# Sadece sayısal kolonları ölçeklendir
pivot_df[num_cols] = scaler.fit_transform(pivot_df[num_cols])

# Kontrol edelim – ilk birkaç satıra bakalım
print(pivot_df[num_cols].head())

# Adım9: Elimizdeki veri seti üzerinden minimum hata ile futbolcuların potansiyel etiketlerini tahmin eden bir makine öğrenmesi modeli
# geliştiriniz. (Roc_auc, f1, precision, recall, accuracy metriklerini yazdırınız.)

#Accuracy: Doğru sınıflandırılan örneklerin oranı.
#Precision: Modelin "highlighted" dediği oyuncuların ne kadarının gerçekten highlighted olduğu.
#Recall: Gerçek highlighted oyuncuların ne kadarını model yakalayabildiği.
#F1 Score: Precision ve Recall’un harmonik ortalaması (dengeyi gösterir).
#ROC AUC: Modelin pozitif ve negatif sınıfları ayırmadaki başarısı (1’e yakın iyi)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report

# Bağımsız değişkenler ve hedef değişken
X = pivot_df[num_cols]
y = pivot_df['potential_label_encoded']

#X: Modelin öğreneceği özellikler (oyuncuların sayısal olarak değerlendirilmiş tüm özellikleri)
#y: Tahmin edilmek istenen hedef değişken (potential_label_encoded, yani average: 0, highlighted: 1)

# Eğitim ve test setlerine ayır (sınıf dengesini koruyarak)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#%80 eğitim, %20 test olarak veri ayrıldı.
#stratify=y: Sınıf oranlarını (0 ve 1) hem eğitim hem test setinde eşit tutar → dengesiz sınıflar varsa önemlidir.
#random_state=42: Sonuçların tekrar üretilebilir olması için.

# Kullanılacak modeller
models = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42)
}

# Sonuçları tutmak için liste
results = []

# Her modeli eğit, test et ve metriklerini hesapla
for name, model in models.items():
    model.fit(X_train, y_train)  # Modeli eğit
    y_pred = model.predict(X_test)  # Sınıf tahminleri
    y_prob = model.predict_proba(X_test)[:, 1]  # Olumlu sınıf (highlighted) için olasılıklar

    # Metrikleri hesapla
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Sınıflandırma raporu yazdır
    print(f"\n{name} Sınıflandırma Raporu:\n")
    print(classification_report(y_test, y_pred))

    # Sonuçları sakla
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "ROC AUC": roc_auc
    })

# Sonuçları DataFrame olarak göster
results_df = pd.DataFrame(results)
print("\nModel Performans Karşılaştırması:\n")
print(results_df)

#Eğer pozitifleri yakalamak (recall) çok önemliyse XGBoost'u düşünebiliriz.
#Genel olarak dengeli ve yüksek performans için RandomForest en iyi tercih gibi görünüyor.


#Adım10: Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz
import matplotlib.pyplot as plt
import seaborn as sns

# Özellik önem düzeyi görselleştirme fonksiyonu
def feature_importance_plot(model, model_name, feature_names, top_n=10):
    """
    Verilen model için en önemli değişkenleri çubuk grafik olarak gösterir.
    
    :param model: Eğitilmiş model (RandomForest, XGBoost, LightGBM)
    :param model_name: Model ismi (str)
    :param feature_names: Özellik isimleri (list)
    :param top_n: Grafikte gösterilecek en önemli n değişken
    """
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
    plt.title(f'{model_name} - En Önemli {top_n} Özellik')
    plt.xlabel('Önem Düzeyi')
    plt.ylabel('Özellik')
    plt.tight_layout()
    plt.show()

# Tüm modeller için çizim
for name, model in models.items():
    feature_importance_plot(model, name, num_cols, top_n=15)
