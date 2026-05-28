import os
import random
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Create directory structure
os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("notebooks", exist_ok=True)
os.makedirs("app", exist_ok=True)

# ---------------------------------------------------------
# TASK 1: SYNTHETIC DATA GENERATION DEFINITIONS & TEMPLATES
# ---------------------------------------------------------

DEPARTMENTS = {
    "Water Supply": {
        "specializations": ["Water main leakage", "Contaminated water", "Low water pressure"],
        "officers": ["Rajesh Kumar", "Sanjay Sharma", "Amit Verma", "Neha Singh", "Priya Patel", "Vijay Yadav", "Deepak Gupta", "Anil Mishra"]
    },
    "Sanitation": {
        "specializations": ["Trash disposal", "Sewer blockage", "Street sweeping"],
        "officers": ["Ramesh Prasad", "Suresh Chandra", "Kiran Devi", "Jyoti Rao", "Rahul Bose", "Rohan Das", "Vikram Sen", "Karan Johar"]
    },
    "Traffic & Roads": {
        "specializations": ["Pothole repair", "Traffic signal malfunction", "Street light outage"],
        "officers": ["Arjun Reddy", "Aditya Nair", "Sandeep Joshi", "Pooja Mehta", "Harish Choudhury", "Sunil Saxena", "Manish Roy"]
    },
    "Public Health": {
        "specializations": ["Pest control", "Food safety violations", "Noise pollution control"],
        "officers": ["Alok Trivedi", "Swati Kapoor", "Rajeshwar Rao", "Meena Iyer", "Vinay Malhotra", "Kavita Joshi", "Aniruddh Sen"]
    },
    "Electricity & Power": {
        "specializations": ["Power line hazard", "Transformer failure", "Electricity meter fault"],
        "officers": ["Rakesh Sharma", "Jitendra Kumar", "Manoj Gupta", "Siddharth Singh", "Nitin Verma", "Abhishek Patel", "Pankaj Mishra"]
    },
    "Environmental Protection": {
        "specializations": ["Illegal dumping", "Industrial emissions", "Deforestation"],
        "officers": ["Devendra Pandey", "Subhash Chandra", "Anjali Roy", "Sunita Deshmukh", "Gaurav Sen", "Pradeep Nair", "Vivek Anand"]
    },
    "Housing & Infrastructure": {
        "specializations": ["Building safety inspections", "Public park maintenance", "Sidewalk repair"],
        "officers": ["Mahendra Singh", "Suresh Prabhu", "Dinesh Karthik", "Ramanathan Iyer", "Kalyan Banerjee", "Ashok Khemka", "Sudarshan Sen"]
    }
}

REGIONS = ["North Zone", "South Zone", "East Zone", "West Zone", "Central Zone"]
LANGUAGES_POOL = ["English", "Spanish", "French", "Hindi", "Chinese"]

# Templates for generating multilingual complaints
TEMPLATES = {
    "Water main leakage": {
        "priority": "High",
        "resolution_range": (1, 3),
        "texts": {
            "English": [
                "There is a major water main burst on {street}. Flooding is spreading quickly on the road.",
                "Water is gushing out of the street near {street}. The entire sidewalk is submerged, please send repair teams.",
                "Massive pipe burst at {street}. Hundreds of gallons of clean water are being wasted and flooding homes."
            ],
            "Spanish": [
                "Hay una gran rotura de tubería de agua en {street}. La inundación se está extendiendo rápidamente en la carretera.",
                "El agua sale a chorros de la calle cerca de {street}. Toda la acera está sumergida, envíen equipos de reparación.",
                "Rotura masiva de tuberías en {street}. Cientos de galones de agua limpia se están desperdiciando e inundando hogares."
            ],
            "French": [
                "Il y a une rupture majeure de canalisation d'eau sur {street}. L'inondation se propage rapidement sur la route.",
                "L'eau jaillit de la rue près de {street}. Tout le trottoir est submergé, veuillez envoyer des équipes de réparation.",
                "Rupture massive de canalisation à {street}. Des centaines de gallons d'eau propre sont gaspillés et inondent les maisons."
            ],
            "Hindi": [
                "{street} पर पानी की एक बड़ी पाइपलाइन फट गई है। सड़क पर बाढ़ तेजी से फैल रही है।",
                "{street} के पास सड़क से पानी बह रहा है। पूरा फुटपाथ डूब गया है, कृपया मरम्मत दल भेजें।",
                "{street} पर पाइपलाइन फटने से सैकड़ों गैलन साफ पानी बर्बाद हो रहा है और घरों में पानी भर रहा है।"
            ],
            "Chinese": [
                "{street}发生主水管爆裂。积水正在马路上迅速蔓延。",
                "{street}附近的马路上正有水涌出。整个人行道都淹没了，请派人修理。",
                "{street}发生严重水管爆裂。成百上千加仑的干净水正在流失并淹没民房。"
            ]
        }
    },
    "Contaminated water": {
        "priority": "High",
        "resolution_range": (1, 3),
        "texts": {
            "English": [
                "The drinking water at {street} is brown and smells like chemicals. It is unsafe to consume.",
                "We have dirty water coming out of our taps on {street}. It has a strange residue, please check the local reservoir.",
                "Multiple residents on {street} report falling sick from tap water. The water color is yellowish and murky."
            ],
            "Spanish": [
                "El agua potable en {street} es de color marrón y huele a químicos. No es segura para consumir.",
                "Nos sale agua sucia de los grifos en {street}. Tiene un residuo extraño, por favor revisen el depósito local.",
                "Varios residentes en {street} informan haberse enfermado por el agua del grifo. El agua es amarillenta y turbia."
            ],
            "French": [
                "L'eau potable sur {street} est marron et sent les produits chimiques. Elle est impropre à la consommation.",
                "De l'eau sale sort de nos robinets sur {street}. Il y a un résidu étrange, veuillez vérifier le réservoir local.",
                "Plusieurs résidents de la rue {street} signalent être tombés malades à cause de l'eau du robinet. L'eau est jaunâtre."
            ],
            "Hindi": [
                "{street} पर पीने का पानी भूरा है और रसायनों जैसी गंध आ रही है। यह पीने के लिए असुरक्षित है।",
                "{street} पर नल से गंदा पानी आ रहा है। इसमें अजीब अवशेष है, कृपया स्थानीय जलाशय की जांच करें।",
                "{street} के कई निवासियों ने नल के पानी से बीमार होने की शिकायत की है। पानी का रंग पीला और मटमैला है।"
            ],
            "Chinese": [
                "{street}的饮用水发褐且有化学品味。饮用很不安全。",
                "我们在{street}的自来水管道里流出脏水。有奇怪的残留物，请检查当地水库。",
                "{street}的多位居民反映因喝自来水而生病。水色发黄且浑浊。"
            ]
        }
    },
    "Low water pressure": {
        "priority": "Low",
        "resolution_range": (8, 15),
        "texts": {
            "English": [
                "The water pressure on {street} has dropped to a trickle. It's difficult to wash dishes or shower.",
                "Extremely low water pressure in our building at {street}. We can barely get water on the top floors.",
                "For the past three days, water flow is very weak at {street}. Please investigate the supply lines."
            ],
            "Spanish": [
                "La presión del agua en {street} ha bajado a un goteo. Es difícil lavar los platos o ducharse.",
                "Presión de agua extremadamente baja en nuestro edificio en {street}. Apenas nos llega agua a los pisos superiores.",
                "Durante los últimos tres días, el flujo de agua es muy débil en {street}. Por favor investiguen las líneas de suministro."
            ],
            "French": [
                "La pression de l'eau sur {street} a chuté à un filet. Il est difficile de faire la vaisselle ou de prendre une douche.",
                "Pression d'eau extrêmement basse dans notre immeuble sur {street}. On peut à peine avoir de l'eau aux étages supérieurs.",
                "Depuis trois jours, le débit d'eau est très faible sur {street}. Veuillez inspecter les conduites d'alimentation."
            ],
            "Hindi": [
                "{street} पर पानी का दबाव बहुत कम हो गया है। बर्तन धोना या स्नान करना मुश्किल है।",
                "{street} पर हमारे भवन में पानी का दबाव बहुत कम है। ऊपरी मंजिलों पर पानी बमुश्किल मिल पाता है।",
                "पिछले तीन दिनों से, {street} पर पानी का प्रवाह बहुत कमजोर है। कृपया आपूर्ति लाइनों की जांच करें।"
            ],
            "Chinese": [
                "{street}的水压已经降到了细流。洗碗或洗澡都很困难。",
                "我们在{street}的大楼水压极低。顶层几乎接不到水。",
                "过去三天来，{street}的水流非常微弱。请调查供水管线。"
            ]
        }
    },
    "Trash disposal": {
        "priority": "Medium",
        "resolution_range": (3, 7),
        "texts": {
            "English": [
                "Garbage hasn't been collected at {street} for over two weeks. The bins are overflowing onto the street.",
                "Smelly piles of garbage on the sidewalk of {street}. It's attracting rats and flies, please send garbage trucks.",
                "Illegal dump of household trash bags on {street}. It blocks pedestrian access and creates unsanitary conditions."
            ],
            "Spanish": [
                "La basura no se ha recogido en {street} durante más de dos semanas. Los contenedores se desbordan en la calle.",
                "Pilas de basura maloliente en la acera de {street}. Está atrayendo ratas y moscas, por favor envíen camiones.",
                "Vertido ilegal de bolsas de basura doméstica en {street}. Bloquea el acceso peatonal y genera insalubridad."
            ],
            "French": [
                "Les ordures n'ont pas été ramassées sur {street} depuis plus de deux semaines. Les bacs débordent sur la rue.",
                "Des tas de déchets malodorants sur le trottoir de {street}. Cela attire les rats, veuillez envoyer des camions.",
                "Dépôt illégal de sacs poubelles sur la rue {street}. Cela bloque le passage et crée des conditions insalubres."
            ],
            "Hindi": [
                "{street} पर दो सप्ताह से अधिक समय से कचरा नहीं उठाया गया है। कचरा डिब्बे सड़क पर ओवरफ्लो हो रहे हैं।",
                "{street} के फुटपाथ पर बदबूदार कचरे का ढेर लगा है। इससे चूहे और मक्खियां आ रही हैं, कृपया कचरा गाड़ी भेजें।",
                "{street} पर घरेलू कचरे के थैलों का अवैध डंपिंग किया गया है। यह पैदल चलने वालों का रास्ता रोकता है।"
            ],
            "Chinese": [
                "{street}已经两周多没收垃圾了。垃圾箱里的垃圾已溢出到马路上了。",
                "{street}人行道上散发恶臭的垃圾堆。引来了老鼠和苍蝇，请派垃圾车来。",
                "{street}发生非法倾倒生活垃圾袋。阻碍行人通行并造成不卫生的状况。"
            ]
        }
    },
    "Sewer blockage": {
        "priority": "High",
        "resolution_range": (1, 4),
        "texts": {
            "English": [
                "Sewer is backing up on {street}. Foul-smelling black waste water is flooding the street and driveways.",
                "The main drainage manhole is overflowing at {street}. Sewage water is bubbling up, creating a health hazard.",
                "Blocked sewer line has caused sewage to backup into basement drains in homes on {street}."
            ],
            "Spanish": [
                "El alcantarillado está retrocediendo en {street}. Aguas residuales negras y malolientes inundan la calle.",
                "La alcantarilla principal se está desbordando en {street}. El agua de alcantarillado brota, creando peligro.",
                "La línea de alcantarillado bloqueada ha hecho que las aguas residuales regresen a los sótanos en {street}."
            ],
            "French": [
                "Les égouts débordent sur {street}. Des eaux usées noires et malodorantes inondent la rue et les allées.",
                "Le regard d'égout principal déborde à {street}. L'eau d'égout remonte, créant un risque sanitaire.",
                "L'obstruction de la conduite d'égout provoque des refoulements d'égout dans les sous-sols sur {street}."
            ],
            "Hindi": [
                "{street} पर सीवर का गंदा पानी बाहर बह रहा है। दुर्गंधयुक्त काला गंदा पानी सड़क और रास्तों में भर रहा है।",
                "{street} पर मुख्य नाला ओवरफ्लो हो रहा है। सीवेज का पानी उबल रहा है, जिससे स्वास्थ्य का खतरा पैदा हो रहा है।",
                "सीवर लाइन बंद होने से {street} पर घरों के बेसमेंट में सीवेज का पानी वापस आ गया है।"
            ],
            "Chinese": [
                "{street}下水道倒灌。恶臭的黑色废水正在淹没马路和车道。",
                "{street}的主排水沙井溢出。污水正在冒泡，造成健康隐患。",
                "下水道管线堵塞导致污水倒灌进{street}住户的地下室排水管。"
            ]
        }
    },
    "Street sweeping": {
        "priority": "Low",
        "resolution_range": (8, 15),
        "texts": {
            "English": [
                "Lots of leaves, dirt, and gravel have accumulated on {street}. The street sweeper hasn't visited in a month.",
                "We need street sweeping at {street}. Debris on the curb is blocking storm gutters.",
                "Accumulated dirt and broken glass on the side of {street}. It's dangerous for bicycle riders."
            ],
            "Spanish": [
                "Se han acumulado muchas hojas, tierra y grava en {street}. La barredora de calles no pasa hace un mes.",
                "Necesitamos limpieza de calles en {street}. Los escombros en el borde de la acera bloquean las alcantarillas.",
                "Tierra acumulada y vidrios rotos al costado de {street}. Es peligroso para los ciclistas."
            ],
            "French": [
                "Beaucoup de feuilles, de saleté et de gravier accumulés sur {street}. La balayeuse n'est pas passée depuis un mois.",
                "Nous avons besoin d'un balayage de rue sur {street}. Les débris bloquent les gouttières.",
                "Accumulation de terre et de verre brisé sur le côté de {street}. C'est dangereux pour les cyclistes."
            ],
            "Hindi": [
                "{street} पर बहुत सारे पत्ते, गंदगी और बजरी जमा हो गए हैं। सड़क सफाई मशीन एक महीने से नहीं आई है।",
                "हमें {street} पर सड़क सफाई की आवश्यकता है। कर्ब पर मलबा गटरों को अवरुद्ध कर रहा है।",
                "{street} के किनारे जमा गंदगी और कांच के टुकड़े। साइकिल चालकों के लिए यह खतरनाक है।"
            ],
            "Chinese": [
                "{street}上堆积了大量的落叶、泥土和砂石。扫路车已经一个月没来了。",
                "我们需要对{street}进行清扫。路边的碎屑正堵塞着雨水槽。",
                "{street}路边积存的泥土和碎玻璃。对骑自行车的人很危险。"
            ]
        }
    },
    "Pothole repair": {
        "priority": "Medium",
        "resolution_range": (4, 10),
        "texts": {
            "English": [
                "There is a deep, dangerous pothole in the middle of {street}. Cars are swerving to avoid it, which is risky.",
                "A large pothole has opened up near {street}. Several vehicles have suffered damaged tires and rims today.",
                "The road surface on {street} is full of cracks and deep potholes. It needs patching before a major accident occurs."
            ],
            "Spanish": [
                "Hay un bache profundo y peligroso en medio de {street}. Los autos se desvían para evitarlo, lo cual es riesgoso.",
                "Se ha abierto un bache grande cerca de {street}. Varios vehículos han sufrido daños en los neumáticos hoy.",
                "La superficie de la carretera en {street} está llena de baches profundos. Necesita parches antes de un accidente."
            ],
            "French": [
                "Il y a un nid-de-poule profond et dangereux au milieu de {street}. Les voitures l'évitent, ce qui est risqué.",
                "Un grand nid-de-poule s'est formé près de {street}. Plusieurs véhicules ont eu des pneus crevés aujourd'hui.",
                "La chaussée sur {street} est pleine de nids-de-poule profonds. Il faut la réparer avant un accident grave."
            ],
            "Hindi": [
                "{street} के बीच में एक गहरा और खतरनाक गड्ढा है। कारें इससे बचने के लिए मुड़ रही हैं, जो कि जोखिम भरा है।",
                "{street} के पास एक बड़ा गड्ढा हो गया है। आज कई वाहनों के टायर और रिम क्षतिग्रस्त हो गए हैं।",
                "{street} पर सड़क की सतह दरारों और गहरे गड्ढों से भरी है। दुर्घटना से पहले मरम्मत की आवश्यकता है।"
            ],
            "Chinese": [
                "{street}中央有一个又深又危险的坑洼。车子为了躲避它而紧急变道，非常危险。",
                "{street}附近出现了一个大坑洼。今天已经有几辆车因此爆胎和车轮受损。",
                "{street}的路面布满了裂缝和深坑。在发生严重事故之前需要进行修补。"
            ]
        }
    },
    "Traffic signal malfunction": {
        "priority": "High",
        "resolution_range": (1, 2),
        "texts": {
            "English": [
                "The traffic lights at the intersection of {street} are completely dead. Traffic is in chaos and accidents are near.",
                "Traffic signal at {street} is stuck on red in all directions. Huge traffic backup, please dispatch officers.",
                "The pedestrian crossing signal is flashing erratically at {street}, causing confusion for kids crossing the street."
            ],
            "Spanish": [
                "Los semáforos en la intersección de {street} están apagados. El tráfico es un caos y hay peligro de choques.",
                "El semáforo en {street} está atascado en rojo en todas las direcciones. Enorme embotellamiento de tráfico.",
                "La señal de cruce peatonal parpadea erráticamente en {street}, causando confusión a los peatones."
            ],
            "French": [
                "Les feux de signalisation à l'intersection de {street} sont en panne. C'est le chaos et les accidents sont proches.",
                "Le feu de signalisation sur {street} est bloqué au rouge. Énorme embouteillage, veuillez envoyer des agents.",
                "Le signal piéton clignote de manière erratique sur {street}, semant la confusion pour les piétons."
            ],
            "Hindi": [
                "{street} के चौराहे पर ट्रैफिक लाइट पूरी तरह से बंद है। यातायात अव्यवस्थित है और दुर्घटना का खतरा है।",
                "{street} पर ट्रैफिक सिग्नल सभी दिशाओं में लाल रंग पर अटका हुआ है। भारी ट्रैफिक जाम, कृपया अधिकारी भेजें।",
                "{street} पर पैदल यात्री क्रॉसिंग सिग्नल गलत तरीके से चमक रहा है, जिससे भ्रम पैदा हो रहा है।"
            ],
            "Chinese": [
                "{street}十字路口的红绿灯完全坏了。交通陷入混乱，险些发生事故。",
                "{street}处的红绿灯在所有方向上都卡在红灯上。车辆严重拥堵，请派员处理。",
                "{street}处的人行道红绿灯闪烁异常，导致过马路的行人和儿童感到困惑。"
            ]
        }
    },
    "Street light outage": {
        "priority": "Low",
        "resolution_range": (5, 12),
        "texts": {
            "English": [
                "A block of street lights on {street} has been out for a week. The road is pitch dark and residents feel unsafe.",
                "The street lamp in front of {street} is flickering continuously. It's very annoying and keeps us awake.",
                "Dark street at {street} due to broken light fixtures. It's a safety hazard for night commuters."
            ],
            "Spanish": [
                "Varias farolas en {street} no funcionan hace una semana. La calle está oscura y es insegura.",
                "La farola frente a {street} parpadea continuamente. Es muy molesto y no nos deja dormir.",
                "Calle oscura en {street} debido a luminarias rotas. Es un peligro para quienes viajan de noche."
            ],
            "French": [
                "Plusieurs lampadaires sur {street} sont éteints depuis une semaine. La route est dans le noir complet.",
                "Le lampadaire devant le {street} clignote en permanence. C'est très agaçant et nous empêche de dormir.",
                "Rue sombre à {street} à cause de luminaires cassés. C'est dangereux pour les usagers de nuit."
            ],
            "Hindi": [
                "{street} पर स्ट्रीट लाइटों का एक पूरा ब्लॉक एक सप्ताह से बंद है। सड़क पर घने अंधेरे से निवासी असुरक्षित महसूस करते हैं।",
                "{street} के सामने स्ट्रीट लैंप लगातार टिमटिमा रहा है। यह बहुत कष्टप्रद है और हमें सोने नहीं देता।",
                "टूटी हुई लाइटों के कारण {street} पर सड़क पर अंधेरा है। यह रात के यात्रियों के लिए एक सुरक्षा खतरा है।"
            ],
            "Chinese": [
                "{street}上的一排路灯坏了一周了。路面漆黑一片，居民感到很不安全。",
                "{street}门前的路灯不停闪烁。非常烦人，让我们无法入睡。",
                "由于照明设备损坏，{street}道路漆黑。对夜间通勤者是一个安全隐患。"
            ]
        }
    },
    "Pest control": {
        "priority": "Medium",
        "resolution_range": (4, 10),
        "texts": {
            "English": [
                "There is a severe rat infestation in the public alley near {street}. They are chewing through trash bins.",
                "We noticed a massive swarm of wasps near the children's play area in the park at {street}.",
                "A plague of cockroaches has spread from the sewer drains to the sidewalk on {street}."
            ],
            "Spanish": [
                "Hay una grave plaga de ratas en el callejón público cerca de {street}. Muerden los cubos de basura.",
                "Notamos un enjambre masivo de avispas cerca del área de juegos infantiles en el parque en {street}.",
                "Una plaga de cucarachas se ha extendido de los desagües del alcantarillado a la acera en {street}."
            ],
            "French": [
                "Il y a une grave infestation de rats dans la ruelle publique près de {street}. Ils rongent les poubelles.",
                "Nous avons remarqué un essaim de guêpes près de l'aire de jeux pour enfants dans le parc à {street}.",
                "Une invasion de cafards s'est propagée des égouts vers le trottoir sur la rue {street}."
            ],
            "Hindi": [
                "{street} के पास सार्वजनिक गली में चूहों का भारी प्रकोप है। वे कचरे के डिब्बों को काट रहे हैं।",
                "हमने {street} के पार्क में बच्चों के खेलने के क्षेत्र के पास ततैया का एक बड़ा झुंड देखा है।",
                "सीवर से लेकर {street} के फुटपाथ तक कॉकरोच फैल गए हैं।"
            ],
            "Chinese": [
                "{street}附近的公共小巷里鼠患严重。老鼠正在啃咬垃圾箱。",
                "我们注意到在{street}的公园里，儿童游乐区附近有大量的黄蜂聚集。",
                "大量的蟑螂已从下水道溢出，蔓延到了{street}的人行道上。"
            ]
        }
    },
    "Food safety violations": {
        "priority": "High",
        "resolution_range": (1, 4),
        "texts": {
            "English": [
                "I saw rats running inside the kitchen of the restaurant on {street}. It's a major hygiene threat.",
                "The grocery store at {street} is selling expired meat and dairy products. The smell is awful.",
                "Multiple people got food poisoning after eating at the food stall at {street}. Please inspect immediately."
            ],
            "Spanish": [
                "Vi ratas corriendo dentro de la cocina del restaurante en {street}. Es una gran amenaza para la higiene.",
                "La tienda de comestibles en {street} vende carne y lácteos vencidos. El olor es horrible.",
                "Varias personas se intoxicaron después de comer en el puesto de {street}. Por favor inspeccionen."
            ],
            "French": [
                "J'ai vu des rats courir dans la cuisine du restaurant sur {street}. C'est une menace majeure pour l'hygiène.",
                "L'épicerie sur {street} vend de la viande et des produits laitiers périmés. L'odeur est affreuse.",
                "Plusieurs personnes ont eu une intoxication après avoir mangé au stand sur {street}. Inspectez d'urgence."
            ],
            "Hindi": [
                "मैंने {street} के रेस्तरां की रसोई के अंदर चूहों को भागते देखा। यह स्वच्छता के लिए एक बड़ा खतरा है।",
                "{street} पर किराना दुकान में एक्सपायरी डेट का मांस और डेयरी उत्पाद बेचे जा रहे हैं। गंध बहुत खराब है।",
                "{street} के फूड स्टॉल पर खाने के बाद कई लोग फूड पॉइजनिंग के शिकार हुए। कृपया तुरंत जांच करें।"
            ],
            "Chinese": [
                "我看到老鼠在{street}餐馆的厨房里跑来跑去。这是重大的卫生隐患。",
                "{street}的杂货店正在出售过期的肉类和奶制品。气味非常难闻。",
                "多人在吃完{street}的小吃摊后发生食物中毒。请立即去检查。"
            ]
        }
    },
    "Noise pollution control": {
        "priority": "Low",
        "resolution_range": (2, 8),
        "texts": {
            "English": [
                "Extremely loud music from the nightclub on {street} past 2 AM. Residents cannot sleep.",
                "Continuous loud construction noise at night on {street}. They are working past the legal curfew.",
                "A resident on {street} is running a noisy generator all day, generating excessive noise vibrations."
            ],
            "Spanish": [
                "Música extremadamente alta del club nocturno en {street} después de las 2 AM. Los vecinos no pueden dormir.",
                "Ruido continuo de construcción por la noche en {street}. Trabajan fuera del horario legal.",
                "Un residente en {street} opera un generador ruidoso todo el día, generando molestas vibraciones."
            ],
            "French": [
                "Musique extrêmement forte de la boîte de nuit sur {street} après 2h du matin. Les voisins ne peuvent pas dormir.",
                "Bruit de chantier continu la nuit sur {street}. Ils travaillent au-delà des heures légales.",
                "Un habitant de la rue {street} fait tourner un générateur bruyant toute la journée, créant des vibrations."
            ],
            "Hindi": [
                "{street} पर नाइट क्लब से रात 2 बजे के बाद बहुत तेज़ संगीत बज रहा है। निवासी सो नहीं पा रहे हैं।",
                "{street} पर रात में लगातार निर्माण कार्य का तेज़ शोर हो रहा है। वे कानूनी समय के बाद भी काम कर रहे हैं।",
                "{street} का एक निवासी पूरे दिन शोर करने वाला जनरेटर चला रहा है, जिससे अत्यधिक शोर कंपन पैदा हो रहा है।"
            ],
            "Chinese": [
                "{street}上的夜总会在凌晨2点后仍在播放极其吵闹的音乐。居民无法入睡。",
                "{street}上夜间持续发生巨大的施工噪音。他们正在超出法定时间工作。",
                "{street}的一位居民整天开着吵闹的发电机，产生了过度的噪音震动。"
            ]
        }
    },
    "Power line hazard": {
        "priority": "High",
        "resolution_range": (1, 2),
        "texts": {
            "English": [
                "A tree branch fell and snapped a power line on {street}. Sparks are flying on the wet pavement, urgent!",
                "There is a downed power cable laying across the street near {street}. It's active and extremely dangerous.",
                "High-voltage electrical wire hanging very low over the sidewalk at {street}. Pedestrians are in danger."
            ],
            "Spanish": [
                "Una rama cayó y rompió un cable eléctrico en {street}. Hay chispas en la acera mojada, ¡urgente!",
                "Hay un cable eléctrico caído en la calle cerca de {street}. Está activo y es extremadamente peligroso.",
                "Cable eléctrico de alta tensión colgando muy bajo sobre la acera en {street}. Los peatones corren peligro."
            ],
            "French": [
                "Une branche d'arbre est tombée et a sectionné un câble électrique sur {street}. Des étincelles jaillissent !",
                "Un câble électrique abattu gît sur la chaussée près de {street}. Il est sous tension et extrêmement dangereux.",
                "Fil électrique haute tension suspendu très bas au-dessus du trottoir à {street}. Les piétons sont en danger."
            ],
            "Hindi": [
                "{street} पर एक पेड़ की शाखा गिरने से बिजली का तार टूट गया है। गीले फुटपाथ पर चिंगारियां निकल रही हैं, तत्काल मदद चाहिए!",
                "{street} के पास सड़क पर बिजली का तार गिरा हुआ है। यह चालू है और बेहद खतरनाक है।",
                "{street} पर फुटपाथ के ऊपर हाई-वोल्टेज बिजली का तार बहुत नीचे लटका हुआ है। राहगीर खतरे में हैं।"
            ],
            "Chinese": [
                "树枝掉落并折断了{street}上的电线。潮湿的地面上正火花四溅，十分紧急！",
                "{street}附近的马路上有一根掉落的电缆。它仍然带电，极其危险。",
                "{street}人行道上方悬挂着一根很低的压电线。行人面临触电危险。"
            ]
        }
    },
    "Transformer failure": {
        "priority": "High",
        "resolution_range": (1, 3),
        "texts": {
            "English": [
                "The electric transformer exploded with a loud bang at {street}. Smoke is rising, and power is out.",
                "A transformer on {street} is leaking oil and humming loudly. Sparks are occasionally coming out.",
                "Power transformer failure at {street} has left the entire neighborhood without electricity and heating."
            ],
            "Spanish": [
                "El transformador eléctrico explotó con un fuerte estallido en {street}. Sale humo y no hay luz.",
                "Un transformador en {street} pierde aceite y zumba con fuerza. Salen chispas ocasionalmente.",
                "Falla del transformador en {street} dejó a todo el vecindario sin electricidad ni calefacción."
            ],
            "French": [
                "Le transformateur électrique a explosé avec un grand bruit à {street}. De la fumée s'élève, plus de courant.",
                "Un transformateur sur {street} fuit de l'huile et grésille fortement. Des étincelles jaillissent parfois.",
                "Panne de transformateur à {street} laissant tout le quartier sans électricité ni chauffage."
            ],
            "Hindi": [
                "{street} पर बिजली का ट्रांसफार्मर जोरदार धमाके के साथ फट गया। धुआं उठ रहा है, और बिजली चली गई है।",
                "{street} पर एक ट्रांसफार्मer से तेल लीक हो रहा है और तेज़ गुनगुनाहट आ रही है। कभी-कभी चिंगारियां निकलती हैं।",
                "{street} पर बिजली ट्रांसफार्मर खराब होने से पूरा इलाका बिना बिजली और हीटिंग के रह गया है।"
            ],
            "Chinese": [
                "{street}的变压器发生爆炸并发出巨响。浓烟升起，电力中断。",
                "{street}上的一个变压器漏油并发出巨大的嗡嗡声。偶尔有火花冒出。",
                "{street}的电力变压器故障导致整个社区停电，供暖中断。"
            ]
        }
    },
    "Electricity meter fault": {
        "priority": "Low",
        "resolution_range": (7, 14),
        "texts": {
            "English": [
                "The electricity meter at my home on {street} is running extremely fast, billing us three times the normal rate.",
                "Our power meter at {street} is completely dead, screen is blank but electricity is still running.",
                "Smart meter at {street} shows error code E-12. Please send a technician to check it."
            ],
            "Spanish": [
                "El medidor de luz de mi casa en {street} corre extremadamente rápido, facturando el triple de lo normal.",
                "Nuestro medidor de energía en {street} está completamente apagado, la pantalla está en blanco.",
                "El medidor inteligente en {street} muestra el código de error E-12. Envíen un técnico por favor."
            ],
            "French": [
                "Le compteur d'électricité de ma maison sur {street} tourne extrêmement vite, triplant notre facture.",
                "Notre compteur d'électricité à {street} est éteint, l'écran est vide mais le courant passe.",
                "Le compteur intelligent à {street} affiche le code d'erreur E-12. Veuillez envoyer un technicien."
            ],
            "Hindi": [
                "{street} पर मेरे घर का बिजली मीटर अत्यधिक तेज़ चल रहा है, जिससे सामान्य से तीन गुना अधिक बिल आ रहा है।",
                "{street} पर हमारा बिजली मीटर पूरी तरह से बंद है, स्क्रीन खाली है लेकिन बिजली अभी भी चल रही है।",
                "{street} पर स्मार्ट मीटर त्रुटि कोड E-12 दिखा रहा है। कृपया जांच के लिए तकनीशियन भेजें।"
            ],
            "Chinese": [
                "我位于{street}家里的电表转得极快，计费是正常的三倍。",
                "我们位于{street}的电表完全坏了，屏幕空白但电力仍然通畅。",
                "{street}的智能电表显示错误代码 E-12。请派技术人员来检查。"
            ]
        }
    },
    "Illegal dumping": {
        "priority": "Medium",
        "resolution_range": (5, 10),
        "texts": {
            "English": [
                "Someone dumped piles of old tires and construction debris on the empty lot at {street}.",
                "Trucks are dumping industrial waste illegally at night near {street}. The chemicals are leaking into soil.",
                "Dumping of electronic appliances and old mattresses on the roadside at {street}."
            ],
            "Spanish": [
                "Alguien arrojó pilas de neumáticos viejos y escombros de construcción en el lote vacío de {street}.",
                "Camiones tiran residuos industriales de noche cerca de {street}. Los químicos se filtran al suelo.",
                "Vertido de electrodomésticos y colchones viejos al costado del camino en {street}."
            ],
            "French": [
                "Quelqu'un a déversé des tas de vieux pneus et de gravats de chantier sur le terrain vague à {street}.",
                "Des camions déversent des déchets industriels la nuit près de {street}. Les produits s'infiltrent dans le sol.",
                "Dépôt d'appareils électroniques et de vieux matelas sur le bord de la route à {street}."
            ],
            "Hindi": [
                "किसी ने {street} के खाली प्लॉट पर पुराने टायरों और निर्माण मलबे का ढेर लगा दिया है।",
                "ट्रक {street} के पास रात में अवैध रूप से औद्योगिक कचरा डाल रहे हैं। रसायन मिट्टी में लीक हो रहे हैं।",
                "{street} पर सड़क किनारे इलेक्ट्रॉनिक उपकरण और पुराने गद्दे फेंके जा रहे हैं।"
            ],
            "Chinese": [
                "有人在{street}的空地上倾倒了成堆的旧轮胎和建筑垃圾。",
                "卡车夜间在{street}附近非法倾倒工业垃圾。化学物质正渗入土壤。",
                "有人在{street}路边倾倒电子电器和旧床垫。"
            ]
        }
    },
    "Industrial emissions": {
        "priority": "High",
        "resolution_range": (2, 5),
        "texts": {
            "English": [
                "The factory near {street} is emitting thick, black chemical smoke. It is hard to breathe in the neighborhood.",
                "Foul sulfur smell coming from the industrial plant at {street}. It's causing headaches for residents.",
                "Illegal chemical discharge from the factory into the local stream near {street}, killing fish."
            ],
            "Spanish": [
                "La fábrica cerca de {street} emite humo químico negro y denso. Es difícil respirar en el vecindario.",
                "Olor fétido a azufre proveniente de la planta industrial en {street}. Provoca dolores de cabeza a los vecinos.",
                "Descarga química ilegal de la fábrica en el arroyo local cerca de {street}, matando peces."
            ],
            "French": [
                "L'usine près de {street} émet une fumée chimique noire et épaisse. Difficile de respirer dans le quartier.",
                "Odeur fétide de soufre provenant de l'usine industrielle à {street}. Cela donne des maux de tête.",
                "Rejet chimique illégal de l'usine dans le ruisseau local près de {street}, tuant les poissons."
            ],
            "Hindi": [
                "{street} के पास की फैक्ट्री से घना, काला रासायनिक धुआं निकल रहा है। पड़ोस में सांस लेना मुश्किल है।",
                "{street} पर औद्योगिक संयंत्र से गंदी सल्फर की गंध आ रही है। इससे निवासियों को सिरदर्द हो रहा है।",
                "{street} के पास की फैक्ट्री से स्थानीय नाले में अवैध रसायन बहाया जा रहा है, जिससे मछलियां मर रही हैं।"
            ],
            "Chinese": [
                "{street}附近的工厂正在排放滚滚黑烟。社区里呼吸困难。",
                "{street}的工业厂区散发出难闻的硫磺味。这导致居民头痛。",
                "工厂向{street}附近的溪流中非法排放化学物质，导致鱼类死亡。"
            ]
        }
    },
    "Deforestation": {
        "priority": "Medium",
        "resolution_range": (3, 8),
        "texts": {
            "English": [
                "Someone is cutting down protected ancient trees in the reserve park near {street} without permission.",
                "Illegal lumbering activities detected at the woodland area near {street}. Multiple trees are felled.",
                "A developer is clearing trees on public land at {street} without environmental permits."
            ],
            "Spanish": [
                "Alguien está talando árboles antiguos protegidos en el parque de reserva cerca de {street} sin permiso.",
                "Tala ilegal detectada en el bosque cerca de {street}. Varios árboles han sido derribados.",
                "Un urbanista está talando árboles en terrenos públicos en {street} sin permisos ambientales."
            ],
            "French": [
                "Quelqu'un abat des arbres anciens protégés dans le parc de réserve près de {street} sans autorisation.",
                "Activités d'abattage illégal détectées dans la zone boisée près de {street}. Plusieurs arbres abattus.",
                "Un promoteur abat des arbres sur un terrain public à {street} sans permis environnemental."
            ],
            "Hindi": [
                "कोई {street} के पास रिजर्व पार्क में बिना अनुमति के संरक्षित प्राचीन पेड़ों को काट रहा है।",
                "{street} के पास वन क्षेत्र में अवैध लकड़ी काटने की गतिविधियां देखी गई हैं। कई पेड़ काटे गए हैं।",
                "एक डेवलपर बिना पर्यावरण परमिट के {street} पर सार्वजनिक भूमि पर पेड़ों को काट रहा है।"
            ],
            "Chinese": [
                "有人在{street}附近的自然保护公园内擅自砍伐受保护的古树。",
                "在{street}附近的林区发现非法砍伐活动。多棵树木被砍倒。",
                "开发商在未获得环保许可的情况下，清除{street}公共土地上的树木。"
            ]
        }
    },
    "Building safety inspections": {
        "priority": "High",
        "resolution_range": (1, 3),
        "texts": {
            "English": [
                "The balcony wall of the older building at {street} has large cracks and looks ready to collapse onto the sidewalk.",
                "Part of a concrete building facade fell onto the road at {street}. The structure looks unsafe.",
                "Deep structural cracking discovered in the retaining wall of the apartment block at {street}."
            ],
            "Spanish": [
                "La pared del balcón del edificio antiguo en {street} tiene grietas grandes y parece a punto de colapsar.",
                "Parte de la fachada de hormigón de un edificio cayó a la calle en {street}. La estructura parece insegura.",
                "Grietas estructurales profundas descubiertas en el muro de contención del bloque de pisos en {street}."
            ],
            "French": [
                "Le mur du balcon de l'ancien immeuble à {street} présente de larges fissures et semble sur le point de s'effondrer.",
                "Une partie de la façade en béton d'un immeuble est tombée sur la route à {street}. La structure semble instable.",
                "Fissures structurelles profondes découvertes dans le mur de soutènement de l'immeuble à {street}."
            ],
            "Hindi": [
                "{street} पर पुरानी इमारत के बालकनी की दीवार में बड़ी दरारें हैं और ऐसा लगता है कि यह फुटपाथ पर गिर जाएगी।",
                "{street} पर कंक्रीट की इमारत के अग्रभाग का एक हिस्सा सड़क पर गिर गया। संरचना असुरक्षित लग रही है।",
                "{street} पर अपार्टमेंट ब्लॉक की रिटेनिंग वॉल में गहरी संरचनात्मक दरारें पाई गई हैं।"
            ],
            "Chinese": [
                "{street}老旧建筑阳台墙体出现巨大裂缝，看起来随时会坍塌到人行道上。",
                "{street}处的混凝土建筑外墙部分脱落掉在马路上。结构看起来很不安全。",
                "{street}公寓大楼的挡土墙发现深层结构裂缝。"
            ]
        }
    },
    "Public park maintenance": {
        "priority": "Low",
        "resolution_range": (8, 15),
        "texts": {
            "English": [
                "The playground equipment at the park on {street} is broken and rusty. It is hazardous for children.",
                "Overgrown weeds and trash piles in the local park at {street}. It looks neglected and abandoned.",
                "Fences are broken and benches are vandalized in the public garden near {street}."
            ],
            "Spanish": [
                "Los juegos del parque infantil en {street} están rotos y oxidados. Es peligroso para los niños.",
                "Maleza alta y basura en el parque local de {street}. Parece descuidado y abandonado.",
                "Las cercas están rotas y los bancos vandalizados en el jardín público cerca de {street}."
            ],
            "French": [
                "Les jeux de l'aire de loisirs sur {street} sont cassés et rouillés. C'est dangereux pour les enfants.",
                "Mauvaises herbes et tas de déchets dans le parc local à {street}. Il semble négligé.",
                "Les clôtures sont cassées et les bancs vandalisés dans le jardin public près de {street}."
            ],
            "Hindi": [
                "{street} के पार्क में बच्चों के झूले टूटे और जंग खाए हुए हैं। यह बच्चों के लिए खतरनाक है।",
                "{street} के स्थानीय पार्क में खरपतवार और कचरे का ढेर लगा है। यह उपेक्षित और परित्यक्त दिखता है।",
                "{street} के पास सार्वजनिक पार्क में बाड़ टूटी हुई हैं और बेंच क्षतिग्रस्त कर दिए गए हैं।"
            ],
            "Chinese": [
                "{street}公园内的游乐场设施损坏且生锈。对儿童玩耍很不安全。",
                "{street}的社区公园里杂草丛生，垃圾堆积。看起来破败不堪，无人打理。",
                "{street}附近的公共花园围栏破损，长椅遭到恶意破坏。"
            ]
        }
    },
    "Sidewalk repair": {
        "priority": "Low",
        "resolution_range": (6, 12),
        "texts": {
            "English": [
                "The concrete sidewalk at {street} is buckled and cracked by tree roots, posing a severe tripping hazard.",
                "A slab of pavement is missing on the sidewalk of {street}, leaving an open muddy hole.",
                "Uneven sidewalk slabs at {street} make it impossible for wheelchair users to navigate safely."
            ],
            "Spanish": [
                "La acera de hormigón en {street} está doblada y agrietada por las raíces, lo que representa peligro de tropezar.",
                "Falta una losa de pavimento en la acera de {street}, dejando un agujero de barro abierto.",
                "Las losas de acera desiguales en {street} hacen imposible que usuarios de sillas de ruedas circulen seguros."
            ],
            "French": [
                "Le trottoir en béton à {street} est déformé par les racines des arbres, posant un risque de chute.",
                "Une dalle de béton manque sur le trottoir de {street}, laissant un trou boueux.",
                "Les dalles de trottoir inégales à {street} rendent l'accès impossible pour les fauteuils roulants."
            ],
            "Hindi": [
                "{street} पर कंक्रीट का फुटपाथ पेड़ की जड़ों से मुड़ गया है और टूट गया है, जिससे गिरने का खतरा है।",
                "{street} के फुटपाथ पर फुटपाथ का एक हिस्सा गायब है, जिससे मिट्टी का गड्ढा बन गया है।",
                "{street} पर असमान फुटपाथ के कारण व्हीलचेयर उपयोगकर्ताओं के लिए सुरक्षित रूप से चलना असंभव हो गया है।"
            ],
            "Chinese": [
                "{street}的水泥人行道被树根拱起开裂，造成严重绊倒隐患。",
                "{street}人行道上缺了一块铺路石板，留下一个敞开的泥坑。",
                "{street}处高低不平的人行道板导致轮椅使用者无法安全通行。"
            ]
        }
    }
}

STREET_NAMES = [
    "MG Road", "Netaji Subhash Chandra Bose Road", "Jawaharlal Nehru Marg", "Dr. B.R. Ambedkar Road",
    "Linking Road", "Brigade Road", "Commercial Street", "Chhatrapati Shivaji Maharaj Marg",
    "Sardar Patel Marg", "Outer Ring Road", "Mall Road", "Chandni Chowk",
    "Lal Bahadur Shastri Marg", "Golf Course Road", "Vittal Mallya Road", "Senapati Bapat Marg",
    "Hazratganj", "Park Street", "SP Road", "Janpath"
]

def generate_synthetic_data():
    print("Generating synthetic datasets...")
    
    # 1. Generate Officers (50 officers)
    officers = []
    officer_counter = 1
    
    for dept, info in DEPARTMENTS.items():
        specs = info["specializations"]
        officer_names = info["officers"]
        
        # Ensure we cover a range of experience, regions, and languages
        for name in officer_names:
            # Randomly select 1-3 languages
            langs = random.sample(LANGUAGES_POOL, k=random.randint(1, 3))
            if "English" not in langs:  # Ensure english coverage mostly
                langs.append("English")
            langs = list(set(langs))
            
            # Specialization
            spec = random.choice(specs)
            
            # Region
            region = random.choice(REGIONS)
            
            # Experience
            experience = random.randint(2, 22)
            
            # Workload (initial random, e.g. 10 to 60)
            workload = random.randint(10, 60)
            
            officers.append({
                "officer_id": f"OFF_{officer_counter:03d}",
                "name": name,
                "department": dept,
                "specialization": spec,
                "languages": str(langs), # Stored as string list
                "experience_years": experience,
                "region": region,
                "workload_score": workload
            })
            officer_counter += 1
            
    # Add filler officers if below 50
    first_names = ["Rajesh", "Amit", "Sanjay", "Vijay", "Anil", "Sunil", "Priya", "Neha", "Deepak", "Ramesh", "Suresh", "Vikram", "Kiran", "Jyoti", "Rahul", "Rohan", "Arjun", "Aditya", "Sandeep", "Pooja"]
    last_names = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Joshi", "Mehta", "Reddy", "Nair", "Rao", "Mishra", "Sen", "Das", "Choudhury", "Bose", "Saxena", "Roy", "Yadav", "Trivedi"]
    
    while len(officers) < 55:
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        if name in [o["name"] for o in officers]:
            continue
        dept = random.choice(list(DEPARTMENTS.keys()))
        spec = random.choice(DEPARTMENTS[dept]["specializations"])
        langs = list(set(random.sample(LANGUAGES_POOL, k=random.randint(1, 2)) + ["English"]))
        region = random.choice(REGIONS)
        
        officers.append({
            "officer_id": f"OFF_{officer_counter:03d}",
            "name": name,
            "department": dept,
            "specialization": spec,
            "languages": str(langs),
            "experience_years": random.randint(1, 20),
            "region": region,
            "workload_score": random.randint(5, 50)
        })
        officer_counter += 1

    officers_df = pd.DataFrame(officers)
    officers_df.to_csv("data/officers.csv", index=False)
    print(f"Generated {len(officers_df)} officers in 'data/officers.csv'.")
    
    # 2. Generate Complaints (500 complaints)
    complaints = []
    start_date = datetime.now() - timedelta(days=365)
    
    for i in range(520):
        # Select random spec and details
        category = random.choice(list(DEPARTMENTS.keys()))
        spec = random.choice(DEPARTMENTS[category]["specializations"])
        temp_info = TEMPLATES[spec]
        
        # Select language
        lang = random.choice(LANGUAGES_POOL)
        
        # Select template text
        text_template = random.choice(temp_info["texts"][lang])
        street = random.choice(STREET_NAMES)
        text = text_template.format(street=street)
        
        # Region / Location
        location = random.choice(REGIONS)
        
        # Priority
        priority = temp_info["priority"]
        # Add a tiny bit of noise/variance to priority to make ML training realistic
        if random.random() < 0.08:
            priority = random.choice(["High", "Medium", "Low"])
            
        # Resolution days (based on priority + some variance)
        min_d, max_d = temp_info["resolution_range"]
        res_days = random.randint(min_d, max_d)
        if priority == "High":
            res_days = max(1, res_days + random.randint(-1, 1))
        elif priority == "Medium":
            res_days = max(2, res_days + random.randint(-2, 2))
        else:
            res_days = max(5, res_days + random.randint(-3, 3))
            
        # Date created
        created_date = start_date + timedelta(
            days=random.randint(0, 360),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        )
        
        complaints.append({
            "complaint_id": f"COM_{i+1:04d}",
            "complaint_text": text,
            "language": lang,
            "category": category,
            "location": location,
            "priority": priority,
            "resolution_days": res_days,
            "created_date": created_date.strftime("%Y-%m-%d %H:%M:%S")
        })

    complaints_df = pd.DataFrame(complaints)
    
    # Assign officers using our routing utility logic (to ensure assigned_officer has logical backing)
    # We will simulate routing for each generated complaint
    from utils.routing import route_complaint, get_officer_embeddings
    
    print("Pre-calculating officer profiles and routing for synthetic dataset...")
    # Temporarily import preprocessing to embed for routing
    from utils.preprocessing import clean_text
    
    # Clean complaint text in df
    complaints_df["clean_text"] = complaints_df["complaint_text"].apply(clean_text)
    
    officer_embeddings = get_officer_embeddings(officers_df)
    
    assigned_officers = []
    for idx, row in complaints_df.iterrows():
        assigned_id, _ = route_complaint(
            complaint_text=row["clean_text"],
            complaint_language=row["language"],
            complaint_location=row["location"],
            officers_df=officers_df,
            officer_embeddings=officer_embeddings,
            top_k=1
        )
        assigned_officers.append(assigned_id)
        
        # Slowly increment officer workload to simulate real dynamic allocation
        officers_df.loc[officers_df["officer_id"] == assigned_id, "workload_score"] += 1.0

    complaints_df["assigned_officer"] = assigned_officers
    
    # Save files
    # Drop clean_text column to keep matching database schema
    complaints_df = complaints_df.drop(columns=["clean_text"])
    complaints_df.to_csv("data/complaints.csv", index=False)
    
    # Save updated officers CSV (now with updated workloads)
    officers_df.to_csv("data/officers.csv", index=False)
    
    print(f"Generated {len(complaints_df)} complaints in 'data/complaints.csv'.")
    print("Synthetic datasets generation finished successfully.")
    
    return complaints_df, officers_df


# ---------------------------------------------------------
# TASK 4 & 5: MODEL TRAINING & METRICS COMPARISON
# ---------------------------------------------------------

def train_and_evaluate_models(complaints_df):
    print("Starting ML Model Training pipeline...")
    from utils.preprocessing import clean_text, generate_embeddings
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_absolute_error, root_mean_squared_error, r2_score
    
    # Preprocess text and generate embeddings
    print("Generating sentence embeddings for training dataset...")
    cleaned_texts = complaints_df["complaint_text"].apply(clean_text).tolist()
    embeddings = generate_embeddings(cleaned_texts) # (N, 384)
    
    # One-hot encode categorical features (category and location)
    print("Preprocessing categorical features...")
    cat_features = complaints_df[["category", "location"]]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['category', 'location'])
        ]
    )
    
    encoded_cats = preprocessor.fit_transform(cat_features)
    
    # Save the pipeline preprocessor for inference time
    joblib.dump(preprocessor, "models/pipeline_preprocessor.pkl")
    print("Fitted categorical preprocessor saved to 'models/pipeline_preprocessor.pkl'.")
    
    # Combine text embeddings and categorical features
    X = np.hstack((embeddings, encoded_cats))
    
    # Targets
    y_priority = complaints_df["priority"]
    y_eta = complaints_df["resolution_days"]
    
    # ---------------------------------------------------------
    # Train Priority Classifier
    # ---------------------------------------------------------
    print("\n--- Training Priority Classifier ---")
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_priority, test_size=0.2, random_state=42)
    
    # Logistic Regression
    lr_clf = LogisticRegression(max_iter=1000, random_state=42)
    lr_clf.fit(X_train_c, y_train_c)
    y_pred_lr = lr_clf.predict(X_test_c)
    
    acc_lr = accuracy_score(y_test_c, y_pred_lr)
    prec_lr, rec_lr, f1_lr, _ = precision_recall_fscore_support(y_test_c, y_pred_lr, average='weighted')
    
    # Random Forest
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train_c, y_train_c)
    y_pred_rf = rf_clf.predict(X_test_c)
    
    acc_rf = accuracy_score(y_test_c, y_pred_rf)
    prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y_test_c, y_pred_rf, average='weighted')
    
    print("Priority Model Performance Comparison:")
    comparison_clf = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1 Score (Weighted)"],
        "Logistic Regression": [acc_lr, prec_lr, rec_lr, f1_lr],
        "Random Forest Classifier": [acc_rf, prec_rf, rec_rf, f1_rf]
    })
    print(comparison_clf.to_string(index=False))
    
    # Choose best and save
    best_clf = "Random Forest Classifier" if f1_rf >= f1_lr else "Logistic Regression"
    print(f"Selecting best classifier model: {best_clf}")
    if best_clf == "Random Forest Classifier":
        joblib.dump(rf_clf, "models/priority_model.pkl")
    else:
        joblib.dump(lr_clf, "models/priority_model.pkl")
    print("Priority Classifier model saved to 'models/priority_model.pkl'.")
    
    # ---------------------------------------------------------
    # Train ETA Regressor
    # ---------------------------------------------------------
    print("\n--- Training ETA Regressor ---")
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_eta, test_size=0.2, random_state=42)
    
    # Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_r, y_train_r)
    y_pred_lin = lin_reg.predict(X_test_r)
    
    mae_lin = mean_absolute_error(y_test_r, y_pred_lin)
    rmse_lin = root_mean_squared_error(y_test_r, y_pred_lin)
    r2_lin = r2_score(y_test_r, y_pred_lin)
    
    # Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_r, y_train_r)
    y_pred_rf_r = rf_reg.predict(X_test_r)
    
    mae_rf = mean_absolute_error(y_test_r, y_pred_rf_r)
    rmse_rf = root_mean_squared_error(y_test_r, y_pred_rf_r)
    r2_rf = r2_score(y_test_r, y_pred_rf_r)
    
    print("ETA Model Performance Comparison:")
    comparison_reg = pd.DataFrame({
        "Metric": ["MAE (Days)", "RMSE (Days)", "R² Score"],
        "Linear Regression": [mae_lin, rmse_lin, r2_lin],
        "Random Forest Regressor": [mae_rf, rmse_rf, r2_rf]
    })
    print(comparison_reg.to_string(index=False))
    
    # Choose best and save
    best_reg = "Random Forest Regressor" if r2_rf >= r2_lin else "Linear Regression"
    print(f"Selecting best regressor model: {best_reg}")
    if best_reg == "Random Forest Regressor":
        joblib.dump(rf_reg, "models/eta_model.pkl")
    else:
        joblib.dump(lin_reg, "models/eta_model.pkl")
    print("ETA Regressor model saved to 'models/eta_model.pkl'.")
    
    # ---------------------------------------------------------
    # TASK 6: FAISS INDEX GENERATION
    # ---------------------------------------------------------
    print("\n--- Generating FAISS Index ---")
    from utils.similarity import FAISSSimilarityIndex
    faiss_index = FAISSSimilarityIndex(dimension=384)
    faiss_index.build_and_add(embeddings)
    faiss_index.save("models/complaints_faiss.index")
    
    # Return metrics for notebook injection
    return {
        "clf_metrics": comparison_clf.to_dict(orient="records"),
        "reg_metrics": comparison_reg.to_dict(orient="records"),
        "best_clf": best_clf,
        "best_reg": best_reg
    }

# Helper to split arrays exactly like train_test_split (avoid dependency bugs)
def train_test_split(X, y, test_split_size=0.2, random_state=42):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_split_size)
    test_idx = indices[:test_size]
    train_idx = indices[test_size:]
    return X[train_idx], X[test_idx], y.iloc[train_idx], y.iloc[test_idx]


# ---------------------------------------------------------
# WRITE JUPYTER NOTEBOOK PROGRAMMATICALLY
# ---------------------------------------------------------

def create_model_training_notebook(metrics):
    print("Generating Jupyter Notebook 'notebooks/model_training.ipynb'...")
    
    # Create the cells structure
    notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# AI/ML Complaint Auto-Routing System - Model Training & Evaluation\n",
                    "This notebook demonstrates the end-to-end data science process of training models to predict:\n",
                    "1. **Complaint Priority** (High/Medium/Low) - Classification Task\n",
                    "2. **Resolution ETA** (in days) - Regression Task\n",
                    "\n",
                    "We use features engineered from multi-lingual complaint texts using Sentence Transformers and concatenated categorical variables."
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Environment Setup & Data Loading"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import joblib\n",
                    "import matplotlib.pyplot as plt\n",
                    "import sys\n",
                    "sys.path.append('..')\n",
                    "\n",
                    "# Load datasets\n",
                    "complaints_df = pd.read_csv('../data/complaints.csv')\n",
                    "officers_df = pd.read_csv('../data/officers.csv')\n",
                    "print(f'Complaints: {len(complaints_df)}, Officers: {len(officers_df)}')\n",
                    "complaints_df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Text Embeddings & Categorical Feature Preprocessing\n",
                    "We will clean the complaint text, extract multilingual Sentence Transformer embeddings, and one-hot encode `category` and `location` columns."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from utils.preprocessing import clean_text, generate_embeddings\n",
                    "from sklearn.compose import ColumnTransformer\n",
                    "from sklearn.preprocessing import OneHotEncoder\n",
                    "\n",
                    "print(\"Cleaning text and generating Sentence Transformer embeddings...\")\n",
                    "cleaned_texts = complaints_df['complaint_text'].apply(clean_text).tolist()\n",
                    "embeddings = generate_embeddings(cleaned_texts)\n",
                    "print(f\"Embeddings shape: {embeddings.shape}\")\n",
                    "\n",
                    "print(\"One-hot encoding category and location...\")\n",
                    "cat_features = complaints_df[['category', 'location']]\n",
                    "preprocessor = ColumnTransformer(\n",
                    "    transformers=[\n",
                    "        ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['category', 'location'])\n",
                    "    ]\n",
                    ")\n",
                    "encoded_cats = preprocessor.fit_transform(cat_features)\n",
                    "print(f\"Categorical features shape: {encoded_cats.shape}\")\n",
                    "\n",
                    "# Combine features\n",
                    "X = np.hstack((embeddings, encoded_cats))\n",
                    "y_priority = complaints_df['priority']\n",
                    "y_eta = complaints_df['resolution_days']\n",
                    "print(f\"Final design matrix X shape: {X.shape}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Priority Prediction Model (Classification)\n",
                    "We train and compare **Logistic Regression** and **Random Forest Classifier** models."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.model_selection import train_test_split\n",
                    "from sklearn.linear_model import LogisticRegression\n",
                    "from sklearn.ensemble import RandomForestClassifier\n",
                    "from sklearn.metrics import classification_report, accuracy_score, confusion_matrix\n",
                    "\n",
                    "X_train, X_test, y_train, y_test = train_test_split(X, y_priority, test_size=0.2, random_state=42)\n",
                    "\n",
                    "# Logistic Regression\n",
                    "lr_clf = LogisticRegression(max_iter=1000, random_state=42)\n",
                    "lr_clf.fit(X_train, y_train)\n",
                    "y_pred_lr = lr_clf.predict(X_test)\n",
                    "\n",
                    "print(\"=== Logistic Regression Report ===\")\n",
                    "print(classification_report(y_test, y_pred_lr))\n",
                    "\n",
                    "# Random Forest Classifier\n",
                    "rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
                    "rf_clf.fit(X_train, y_train)\n",
                    "y_pred_rf = rf_clf.predict(X_test)\n",
                    "\n",
                    "print(\"=== Random Forest Classifier Report ===\")\n",
                    "print(classification_report(y_test, y_pred_rf))"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### Priority Model Comparison Summary\n",
                    "Based on our training run, here are the evaluation metrics comparison on the test set:\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Print precalculated comparison\n",
                    "clf_data = " + json.dumps(metrics["clf_metrics"], indent=4) + "\n",
                    "comparison_clf = pd.DataFrame(clf_data)\n",
                    "comparison_clf"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. ETA Prediction Model (Regression)\n",
                    "We train and compare **Linear Regression** and **Random Forest Regressor** models."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from sklearn.linear_model import LinearRegression\n",
                    "from sklearn.ensemble import RandomForestRegressor\n",
                    "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                    "\n",
                    "X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_eta, test_size=0.2, random_state=42)\n",
                    "\n",
                    "# Linear Regression\n",
                    "lin_reg = LinearRegression()\n",
                    "lin_reg.fit(X_train_r, y_train_r)\n",
                    "y_pred_lin = lin_reg.predict(X_test_r)\n",
                    "print(\"=== Linear Regression Metrics ===\")\n",
                    "print(f\"MAE: {mean_absolute_error(y_test_r, y_pred_lin):.3f} days\")\n",
                    "print(f\"RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_lin)):.3f} days\")\n",
                    "print(f\"R² Score: {r2_score(y_test_r, y_pred_lin):.3f}\")\n",
                    "\n",
                    "# Random Forest Regressor\n",
                    "rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)\n",
                    "rf_reg.fit(X_train_r, y_train_r)\n",
                    "y_pred_rf_r = rf_reg.predict(X_test_r)\n",
                    "print(\"\\n=== Random Forest Regressor Metrics ===\")\n",
                    "print(f\"MAE: {mean_absolute_error(y_test_r, y_pred_rf_r):.3f} days\")\n",
                    "print(f\"RMSE: {np.sqrt(mean_squared_error(y_test_r, y_pred_rf_r)):.3f} days\")\n",
                    "print(f\"R² Score: {r2_score(y_test_r, y_pred_rf_r):.3f}\")"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "### ETA Model Comparison Summary\n",
                    "Based on our training run, here are the evaluation metrics comparison on the test set:\n"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "reg_data = " + json.dumps(metrics["reg_metrics"], indent=4) + "\n",
                    "comparison_reg = pd.DataFrame(reg_data)\n",
                    "comparison_reg"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. Saving the Best Models & FAISS Index\n",
                    "We serialize the best models using Joblib and construct the FAISS semantic index for complaints."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Save final models (this was already performed in setup_and_train.py)\n",
                    "print(f\"Selected Priority Classifier: {metrics['best_clf']}\")\n",
                    "print(f\"Selected ETA Regressor: {metrics['best_reg']}\")\n",
                    "\n",
                    "# Verify they can load\n",
                    "loaded_clf = joblib.load('../models/priority_model.pkl')\n",
                    "loaded_reg = joblib.load('../models/eta_model.pkl')\n",
                    "loaded_pre = joblib.load('../models/pipeline_preprocessor.pkl')\n",
                    "print(\"All model artifacts successfully loaded and ready for production inference!\")"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    with open("notebooks/model_training.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=2)
    print("Jupyter Notebook generated successfully in 'notebooks/model_training.ipynb'.")

# ---------------------------------------------------------
# EXECUTION ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    print("=========================================================")
    print("Starting Complaint Routing System Setup & Training Script")
    print("=========================================================")
    
    # 1. Generate Data
    complaints_df, officers_df = generate_synthetic_data()
    
    # 2. Train Models and evaluate
    metrics = train_and_evaluate_models(complaints_df)
    
    # 3. Write Notebook
    create_model_training_notebook(metrics)
    
    print("\n=========================================================")
    print("Setup and Training completed successfully!")
    print("All models, preprocessors, indices, and notebooks saved.")
    print("=========================================================")
