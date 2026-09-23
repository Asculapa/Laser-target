"""The story game in English: every word it shows or says.

To translate the story, copy this file, change the words, and add the new
module to LANGUAGES in story_script.py; tools/story_audio.py then needs a
voice for it. LINES are spoken; their keys are the ids in story_script.py,
and each is also the name of its recording. UI is everything else on screen.
"""

CODE = "en"
NAME = "English"                # the language, as it calls itself
TITLE = "The Last Light of Lantern Rock"
NAMES = {"narrator": "", "maren": "MAREN", "pip": "PIP", "gloam": "THE GLOAM"}
CHAPTERS = (
    "Sparks on the Wind",
    "What the Dark Wants",
    "Ships in the Dark",
    "The Keepers' Stars",
    "The Heart of the Storm",
)

LINES = {
    "title": "The Last Light of Lantern Rock.",
    "c1i01": (
        "For a hundred years, the lamp on Lantern Rock has never gone dark. Tonight, the "
        "storm came to change that."),
    "c1i02": (
        "One strike of lightning. The great lamp burst like a dropped jar of stars, and its "
        "light blew out across the night."),
    "c1i03": (
        "Apprentice! Up here! Mind the glass - and mind my ankle, I have twisted it badly on "
        "the stairs."),
    "c1i04": (
        "Take this. The keeper's shard, the heart of the old lens. It only lights for a "
        "steady hand."),
    "c1i05": (
        "The lamp is not dead, only scattered. Those sparks on the wind are every piece of "
        "it. Catch them, before the storm carries them out to sea."),
    "c1i06": "You raise the shard. A thin bright line leaps from it into the dark.",
    "c1o01": "Ow! Careful! You caught me right in the middle of a very good gust.",
    "c1o02": "The last spark you gathered is looking at you. Sparks, as a rule, do not do that.",
    "c1o03": (
        "I am Pip. I was the bright bit, right at the front of the lamp. You are the new "
        "keeper? You are shorter than the old one."),
    "c1o04": (
        "A talking spark. Forty years I have kept this light, and it never said one word to "
        "me."),
    "c1o05": "You never asked.",
    "c1o06": (
        "The sparks settle in the lamp's glass cage: a small warm glow in a very large dark. "
        "And in the dark, something has noticed."),
    "c1f": "Too many lost to the wind! Again, apprentice - a steady hand!",
    "c2i01": "It comes up the tower stairs without a sound: a grey, cold fog, with too many eyes.",
    "c2i02": (
        "The Gloam. My grandmother told me of it. It is the dark between the stars, and it "
        "has hated this lamp since the day it was lit."),
    "c2i03": (
        "A hundred years of glare. A hundred years I could not cross this water. And now your"
        " lamp is a jar of crumbs."),
    "c2i04": (
        "Crumbs! I am not crumbs! Keeper, it eats light. If those little fog things reach the"
        " cage, they will swallow us one by one."),
    "c2i05": (
        "Light is the one thing it cannot bear. Burn them back with the shard. But leave the "
        "glow moths be - they are on our side, and always were."),
    "c2o01": (
        "The last gloamling curls up like burnt paper and is gone. The fog slides back down "
        "the stairs."),
    "c2o02": (
        "Keep your crumbs, little keeper. There are other lights tonight. Smaller ones. "
        "Further from home."),
    "c2o03": "What does it mean, other lights?",
    "c2o04": (
        "Oh, no. The fishing fleet. Nine boats went out this morning - and without the lamp, "
        "they cannot see the reef."),
    "c2f": "They got us! No, wait - I am still here. Quick, again, before it notices!",
    "c3i01": (
        "From the gallery rail you can see them: small lanterns on black water, coming home "
        "the way they always have, steering for a light that is not there."),
    "c3i02": (
        "The Teeth, we call that reef. There is one safe channel through it, and the lamp has"
        " always pointed the way."),
    "c3i03": "We are not a lamp yet. We are one keeper, and a very thin beam.",
    "c3i04": (
        "Then be a thin lamp. Hold your light on a boat until its skipper sees you and turns "
        "for the channel. Then find the next one. Go!"),
    "c3i05": "Look away, little boats. There is nothing here. Nothing at all.",
    "c3o01": (
        "One by one the boats slip through the channel and bump against the harbour wall. "
        "Somebody down there is cheering. Somebody else is crying, the good kind."),
    "c3o02": (
        "Did you hear that? They cheered for us. Well, for you. But I helped. I glowed "
        "encouragingly."),
    "c3o03": (
        "That was keeper's work. But you cannot hold a lamp in your hand for ever. We must "
        "relight the real one."),
    "c3o04": (
        "And for that, the lens must be set by the stars, the way the first keepers did it. "
        "Help me up to the chart room."),
    "c3f": "The reef has them... No. I will not have it. Again - and hold the light steadier!",
    "c4i01": (
        "The chart room has no roof, only sky. The storm has torn a hole in the clouds, and "
        "through it the stars look down, waiting."),
    "c4i02": (
        "Every keeper learns three constellations: the Boat, the Gull, and the Lamp. Trace "
        "them with the shard, star by star, in the right order, and the lens will turn to "
        "meet them."),
    "c4i03": (
        "I know these! I have been looking at them for a hundred years. Mostly because I "
        "could not look at anything else."),
    "c4i04": (
        "Watch the pattern first. Then draw it back. The sky is patient - but the Gloam is "
        "not."),
    "c4o01": (
        "As the last star answers, something deep in the tower goes clunk. The great lens, "
        "taller than you are, swings round and settles."),
    "c4o02": "There. Set true. All it wants now is its light back.",
    "c4o03": (
        "That would be us. Keeper... I am a little frightened. When we go back into the lamp,"
        " I think I stop being me. I think I just become bright."),
    "c4o04": "Before you can answer, the stars go out. All of them. All at once.",
    "c4f": "The lens has slipped. Never mind. Breathe, watch the pattern, and try again.",
    "c5i01": (
        "The Gloam has stopped creeping. It rises around the tower like a wave that has "
        "decided not to fall, and in the middle of it, one great eye opens."),
    "c5i02": (
        "Enough. I was old when the sea was new. I will not be sent away by a child with a "
        "splinter of glass."),
    "c5i03": (
        "Do not listen to it! It is only dark, and dark has never once beaten a lit lamp. "
        "Find where it is thin, where the light shows through, and strike there!"),
    "c5i04": (
        "Keeper! Whatever happens in the lamp - I would rather be bright with you than safe "
        "in a jar. Come on. Let us go and be a lighthouse!"),
    "c5o01": (
        "The eye closes. For one breath there is nothing at all. Then the lamp of Lantern "
        "Rock catches - and a beam as wide as a road swings out across the sea."),
    "c5o02": "There will be other nights...",
    "c5o03": "There will. And there will be a keeper for every one of them.",
    "c5o04": (
        "The fog burns off the water like breath off a window. Far below, the fleet lies safe"
        " at the harbour wall, and the first pale line of morning is showing in the east."),
    "c5o05": (
        "I am too old for those stairs, and you have just done in one night what I trained "
        "forty years for. The lamp is yours... Keeper."),
    "c5o06": (
        "You look up at the light. It turns, steady and enormous, exactly as it should. And "
        "then, just once, just as it passes you, it flickers."),
    "c5o07": "Still me! Told you. Go to bed, keeper. I have got this.",
    "c5f": "Not dark yet! Get up, keeper! Again!",
}

UI = {
    # the chapter list and the scenes
    "subtitle": "a story in five chapters, told with a laser",
    "locked": "locked",
    "begin": "begin",
    "choose": "hold the laser on a chapter, or press {keys}      ESC back",
    "chapter": "Chapter {n}  -  {title}",
    "skip": "skip >>",
    "next": "flash the laser here, or SPACE: next line      ENTER: skip the scene",
    "complete": "CHAPTER {n} COMPLETE",
    "score": "score {score}",
    "new_best": "   -   a new best",
    "best": "   (best {best})",
    "hits": "{hits} hits   best streak {combo}",
    "continue": "continue",
    "button": "hold the laser on the button, or press SPACE",
    "falters": "THE LIGHT FALTERS",
    "again": "try again",
    "end": "THE END",
    "total": "score {score}     {stars} of {of} stars",
    "replay": "every chapter can be played again for a better result",
    "end_keys": "G chapters    ESC back to tracking",
    "paused": "PAUSED",
    "resume": "P to resume    ESC to quit",
    # every chapter
    "hud_score": "SCORE {score}",
    "streak": "x{x}  ({n} in a row)",
    "miss": "miss",
    # 1
    "hint1": "catch the sparks before the wind takes them",
    "goal1": "catch {need} sparks - do not let {lives} blow away",
    "sparks": "sparks {a} / {b}",
    "blown": "blown away",
    # 2
    "hint2": "burn the gloamlings back before they reach the cage",
    "goal2": "leave the green moths alone - they are friends",
    "wave": "wave {a} / {b}",
    "wave_banner": "WAVE {n}",
    "friend": "a friend!  -25",
    "moths": "Not the moths!",
    "split": "split",
    "spark_out": "a spark goes out",
    # 3
    "hint3": "hold your light on a boat until it turns green",
    "goal3": "bring the fleet through the gap in the reef - lose no more than 2",
    "home": "home {a} / {b}",
    "on_course": "on course",
    "reef": "on the reef",
    # 4
    "hint4": "watch the stars light up, then light them in the same order",
    "goal4": "a wrong star costs a spark, and the pattern is shown again",
    "sky1": "THE BOAT",
    "sky2": "THE GULL",
    "sky3": "THE LAMP",
    "sky": "{name}   {a} / {b}",
    "watch": "{name}  -  watch",
    "trace": "now trace it:  star {a} of {b}",
    "not_that": "not that one",
    # 5
    "hint5": "strike the bright knots as they open",
    "goal5": "and keep the gloamlings off the lamp",
    "boss": "THE GLOAM",
    "taunt1": "YOU ARE ONE SMALL LIGHT",
    "taunt2": "I AM THE WHOLE NIGHT",
    "eye": "NOW - HOLD THE LIGHT ON ITS EYE",
    "lit": "THE LAMP IS LIT",
}
