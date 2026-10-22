"""What the Wild West's announcer says, in Ukrainian.

Nothing here is shown on screen - the banners say it in their own words - so
these are written to be heard: short, loud, the way a man on a porch would
call it, with the punctuation that shapes how they are said. As in the
story, the player is never given a gender: no "ковбою", no past tense. A call that
comes up again and again has variants ("hit", "hit.2", ...), and one of them
is picked each time, so the game does not nag in one tone.

Each line is spoken in one of two deliveries (mood()): "call", shouted over
the street, or "calm", for the tutorial, where a child has to follow what is
said. Recorded by tools/west_voice.py into assets/west/voice/; a line whose
text changes here is played without sound until it is recorded again.
STRESS tells the speech model where to stress the words it gets wrong.
"""

LINES = {
    # the streets, as you come into each
    "street0": "Головна вулиця! Тримай револьвер напоготові.",
    "street1": "Крамниця! Ці ховаються за прилавками.",
    "street2": "Заїзний двір! Тут бігають — цілься наперед.",
    "street3": "Дощаний тротуар!",
    "street4": "Готель Генка! Повно вікон — повно бандитів.",
    "street5": "Каліко! Остання вулиця!",
    # the fights
    "showdown": "Ватажок банди виходить на дуель!",
    "showdown.2": "А ось і ватажок! Дуель!",
    "draw": "Стріляй!",
    "draw.2": "Зараз!",
    "noon": "Рівно опівдні! Хто швидший?",
    "noon_win0": "Дуель виграє перший гравець!",
    "noon_win1": "Дуель виграє другий гравець!",
    "clear": "Вулиця чиста! Рушаймо далі.",
    "clear.2": "Чисто! Жодного бандита.",
    "safe": "Місто в безпеці! Дякуємо тобі!",
    # something to shoot for
    "coin": "Золото! Подвійні очки!",
    "star": "Зірка шерифа — ще одне життя!",
    # what went wrong
    "hit": "Ох! Тебе зачепило!",
    "hit.2": "Обережно! Він стріляв першим!",
    "hit.3": "Влучили! Тримайся!",
    "townsfolk": "Гей! Це ж мирний житель!",
    "townsfolk.2": "Не в мирних! Цілься в бандитів!",
    "early": "Зарано! Чекай сигналу!",
    "slow": "Запізно... Він був швидший.",
    "reload": "Барабан порожній! Перезаряди!",
    "lastlife": "Останнє життя! Тепер обережно...",
    "gameover": "Цього разу банда взяла гору...",
    # the tutorial
    "lesson1": "Бандит вихопив револьвер. Стріляй у нього!",
    "lesson2": "А це мирний житель. Не стріляй... просто дочекайся, поки він піде.",
    "lesson3": "Набої скінчилися. Стріляй у смугу перезарядки — внизу екрана.",
    "lesson4": "А цей стріляє у відповідь. Випереди його!",
    "again": "Нічого. Спробуй ще раз.",
    "good": "Молодець!",
    "good.2": "Точно в ціль!",
    "trained": "Навчання завершено. А тепер — справжнє місто!",
}

# Words the speech model stresses wrongly here: a "+" after the stressed vowel.
STRESS = {
    "нічого": "нічо+го",       # "never mind", not "nothing to"
    "виходить": "вихо+дить",   # is coming out, not will
    "каліко": "Ка+ліко",       # Calico, the town
}

# How each line is said; anything not here is a call.
CALM = {"lesson1", "lesson2", "lesson3", "lesson4", "again", "good", "good.2",
        "trained", "slow", "gameover"}


def mood(key: str) -> str:
    return "calm" if key in CALM else "call"


def variants(key: str) -> list:
    """The line and its variants: "hit", "hit.2", "hit.3"..."""
    return [k for k in LINES if k == key or k.startswith(key + ".")]
