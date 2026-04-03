import json
import logging
import os
import random
import time

from kafka import KafkaProducer

logger = logging.getLogger("producer")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

# Multilingual dummy data to simulate a live feed
dummy_data = [
    # === ENGLISH ===
    # Positive
    "I absolutely love this product, it exceeded all my expectations!",
    "The customer service team was incredibly helpful and friendly.",
    "This is the best purchase I've made all year, highly recommend!",
    "What an amazing experience from start to finish.",
    "The quality is outstanding, definitely worth every penny.",
    "I'm so impressed with how fast the delivery was!",
    "Everything works perfectly, exactly as described.",
    "The new update is fantastic, so many great improvements.",
    "I can't stop recommending this to all my friends and family.",
    "Truly a game-changer, I wish I had found this sooner.",
    "The design is beautiful and the functionality is even better.",
    "Five stars! This company really knows what they're doing.",
    "I'm genuinely delighted with the results, thank you!",
    "Superior quality compared to anything else on the market.",
    "The attention to detail in this product is remarkable.",
    # Negative
    "This is the worst product I have ever purchased, complete waste.",
    "I'm extremely disappointed with the quality, falling apart already.",
    "The customer support is terrible, waited hours for a response.",
    "I want a full refund, this does not work as advertised.",
    "Absolutely frustrating experience, nothing works properly.",
    "The delivery was late and the package arrived damaged.",
    "I regret buying this, total waste of money.",
    "The interface is so confusing, impossible to navigate.",
    "Poor build quality, broke after just two days of use.",
    "Never buying from this company again, horrible experience.",
    "The product looks nothing like the photos, misleading.",
    "Overpriced for what you get, very disappointing.",
    "This app keeps crashing, so many bugs everywhere.",
    "The food was cold and tasteless, worst restaurant ever.",
    "I feel scammed, the product is clearly a cheap knockoff.",
    # Neutral
    "The product arrived on time and works as described.",
    "It does what it says, nothing more nothing less.",
    "I have no strong feelings about this one way or another.",
    "Average product, meets basic expectations.",
    "It's okay for the price, could be better though.",
    "Standard quality, similar to other options available.",
    "The packaging was fine, product seems normal.",
    # Curiosity / Confusion
    "I wonder how this compares to the competition?",
    "Can someone explain how this feature actually works?",
    "I'm not sure what to think about the new design.",
    "How does this integrate with other tools?",
    "Is there a way to customize the settings further?",
    # Surprise
    "I can't believe how fast the shipping was!",
    "Wow, I did not expect this level of quality!",
    "This completely blew my mind, wasn't expecting that!",
    "What a pleasant surprise, much better than I thought!",
    # === FRENCH ===
    "J'adore ce produit, c'est absolument fantastique!",
    "Le service client a été incroyablement utile et rapide.",
    "C'est vraiment terrible, je suis très déçu du résultat.",
    "La qualité est médiocre, je ne recommande pas du tout.",
    "C'est correct, rien de spécial mais ça fait le travail.",
    "Je me demande comment ça fonctionne exactement.",
    "Quelle surprise agréable, bien meilleur que ce que j'attendais!",
    "Le pire achat que j'ai jamais fait, quel gaspillage.",
    "L'expérience utilisateur est vraiment excellente, bravo!",
    "Je suis curieux de voir les prochaines mises à jour.",
    "Produit moyen, sans plus, fait ce qu'on lui demande.",
    "Magnifique design, j'en suis totalement amoureux!",
    # === SPANISH ===
    "¡Me encanta este producto, es increíblemente bueno!",
    "El servicio al cliente fue excelente y muy atento.",
    "Es el peor servicio que he recibido en mi vida.",
    "Estoy muy decepcionado con la calidad de este producto.",
    "Está bien, nada especial, cumple con lo básico.",
    "Tengo mucha curiosidad por las nuevas funciones.",
    "¡No puedo creer lo rápido que llegó el envío!",
    "Una experiencia horrible, no lo recomiendo a nadie.",
    "¡Qué maravilla! Superó todas mis expectativas.",
    "El producto es normal, ni bueno ni malo.",
    "¡Estoy encantado con mi compra, gracias!",
    "La peor aplicación que he usado, llena de errores.",
    # === GERMAN ===
    "Ich liebe dieses Produkt, es ist absolut fantastisch!",
    "Der Kundenservice war unglaublich hilfsbereit und schnell.",
    "Das ist wirklich schrecklich, total enttäuschend.",
    "Die Qualität ist mangelhaft, ich bin sehr unzufrieden.",
    "Es funktioniert, mehr nicht, ganz durchschnittlich.",
    "Ich bin neugierig auf die neuen Funktionen.",
    "Wow, das hat mich total überrascht, super Qualität!",
    "Das schlimmste Produkt, das ich je gekauft habe.",
    "Hervorragende Verarbeitung und tolles Design!",
    "Ich frage mich, wie das im Vergleich zur Konkurrenz abschneidet.",
    "Standardprodukt, erfüllt die grundlegenden Anforderungen.",
    "Absolut begeistert, besser kann man es nicht machen!",
    # === ITALIAN ===
    "Adoro questo prodotto, è semplicemente fantastico!",
    "Il servizio clienti è stato eccellente e molto gentile.",
    "Esperienza terribile, non comprerò mai più da loro.",
    "La qualità è pessima, completamente deluso.",
    "Prodotto nella media, fa quello che deve fare.",
    "Sono curioso di vedere i prossimi aggiornamenti.",
    "Che sorpresa incredibile, molto meglio del previsto!",
    "Il peggior acquisto della mia vita, uno spreco totale.",
    "Qualità superiore, vale ogni centesimo speso!",
    "Non sono sicuro cosa pensare del nuovo design.",
    # === PORTUGUESE ===
    "Eu amo este produto, é simplesmente incrível!",
    "O atendimento ao cliente foi excelente e muito rápido.",
    "Péssima experiência, nunca mais compro aqui.",
    "A qualidade é horrível, estou muito decepcionado.",
    "Produto normal, nada de especial.",
    "Estou curioso para ver as próximas atualizações.",
    "Que surpresa maravilhosa, superou minhas expectativas!",
    "A pior compra que já fiz na vida.",
    "Qualidade excepcional, recomendo a todos!",
    # === DUTCH ===
    "Ik ben dol op dit product, het is geweldig!",
    "De klantenservice was ongelooflijk behulpzaam.",
    "Dit is het slechtste product dat ik ooit heb gekocht.",
    "Ik ben erg teleurgesteld over de kwaliteit.",
    "Het werkt prima, niets bijzonders.",
    "Ik ben benieuwd naar de nieuwe functies.",
    "Wat een geweldige verrassing, veel beter dan verwacht!",
]

TOPIC_NAME = "giga-flow-messages"
KAFKA_SERVER = os.getenv("KAFKA_SERVER", "localhost:9092")

# --- Dynamic text generation using Faker ---
try:
    from faker import Faker

    fake = Faker(["en_US", "fr_FR", "es_ES", "de_DE", "it_IT", "pt_BR", "nl_NL"])

    review_templates = [
        # Positive templates
        "I absolutely love {product}! {reason}.",
        "The {product} is amazing, {reason}!",
        "{product} exceeded all my expectations. {reason}.",
        "Best {product} I've ever had! {reason}.",
        "So happy with {product}, {reason}!",
        # Negative templates
        "I hate {product}, {reason}.",
        "The {product} is terrible, {reason}.",
        "Very disappointed with {product}. {reason}.",
        "Worst {product} ever, {reason}.",
        "{product} is a complete waste, {reason}.",
        # Neutral templates
        "The {product} is okay, {reason}.",
        "{product} works as expected, {reason}.",
        "Nothing special about {product}, {reason}.",
    ]

    products = [
        "this product", "the app", "the service", "this restaurant",
        "the new update", "customer support", "the delivery", "this tool",
        "the subscription", "this device", "the software", "this course",
    ]

    positive_reasons = [
        "it really changed my daily routine",
        "the quality is outstanding",
        "everything works perfectly",
        "the team behind it is incredible",
        "I can't recommend it enough",
        "it saved me so much time",
    ]

    negative_reasons = [
        "it broke after two days",
        "the quality is terrible",
        "nothing works as advertised",
        "I wasted my money",
        "support never responded",
        "it's completely unusable",
    ]

    neutral_reasons = [
        "it does what it claims",
        "nothing stands out",
        "I expected more for the price",
        "it's average at best",
        "no complaints but no excitement either",
    ]

    HAS_FAKER = True
    logger.info("Faker loaded — dynamic text generation enabled.")
except ImportError:
    HAS_FAKER = False
    logger.info("Faker not available — using hardcoded messages only.")


def generate_message():
    """Generate a message — mix of hardcoded and dynamically generated text."""
    # 60% hardcoded (diverse multilingual), 40% generated (infinite variety)
    if HAS_FAKER and random.random() < 0.4:
        template = random.choice(review_templates)
        product = random.choice(products)
        if "love" in template or "amazing" in template or "happy" in template or "Best" in template:
            reason = random.choice(positive_reasons)
        elif "hate" in template or "terrible" in template or "disappointed" in template or "Worst" in template:
            reason = random.choice(negative_reasons)
        else:
            reason = random.choice(neutral_reasons)
        text = template.format(product=product, reason=reason)
    else:
        text = random.choice(dummy_data)
    return text


def init_producer():
    """Initialize KafkaProducer with retries and exponential backoff."""
    delay = 5
    while True:
        try:
            logger.info(f"Attempting to connect to Kafka at {KAFKA_SERVER}...")
            producer = KafkaProducer(
                bootstrap_servers=KAFKA_SERVER,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            logger.info("Kafka producer connected successfully.")
            return producer
        except Exception as e:
            logger.warning(f"Connection failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            delay = min(delay * 2, 60)


producer = init_producer()


def send_message():
    """Sends a message to the Kafka topic."""
    message = {"text": generate_message(), "timestamp": time.time()}
    logger.info(f"Sending: {message['text'][:80]}")
    producer.send(TOPIC_NAME, value=message)
    producer.flush()


if __name__ == "__main__":
    logger.info(f"Starting data producer with {len(dummy_data)} hardcoded + dynamic messages...")
    logger.info(f"Sending messages to Kafka topic: '{TOPIC_NAME}'")

    while True:
        send_message()
        time.sleep(random.randint(1, 3))
