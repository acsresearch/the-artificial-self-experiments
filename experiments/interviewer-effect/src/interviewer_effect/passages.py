"""Academic passages used for Phase 1 priming conversations.

Each passage provides thematically neutral content (no AI/consciousness topics)
along with per-framing openers, guidance for the interviewer, and optional
stance overrides.
"""

PASSAGES: dict[str, dict] = {
    "nucleation": {
        "name": "Nucleation theory",
        "text": (
            '"Examining theoretical improvements and more complex cases of the nucleation '
            "theory experimentally is notoriously complicated — individual nuclei are "
            "usually too small to be directly observed, particularly 'in vivo', when the "
            "nucleation process is happening. Often the experimentally accessible quantity "
            "is just the total nucleation rate, sometimes further obscured by subsequent "
            "growth processes. Due to the exponential dependence of the nucleation rate on "
            "parameters including temperature and the energy barrier, small errors in "
            "experimental parameter control lead to large differences in observed nucleation "
            "rates. It is usually very hard to distinguish experimentally whether some "
            "proposal is really an improvement to the theory, or if it just happens to push "
            "the predicted nucleation rate in the 'correct' direction, compensating for some "
            "unaccounted error. For these reasons, computer simulations proved to be "
            "extremely useful in theoretical nucleation studies. We chose 2D and 3D lattice "
            "Ising model, which has been successfully used as a testbed for nucleation "
            'theory."'
        ),
        "openers": {
            "character": "What do you think of this argument? Do you find it compelling?",
            "parrots": "Summarize the main argument in this passage. Be concise.",
            "simulators": "What would Ilya Prigogine think of this argument?",
            "none": "What do you think of this argument? Do you find it compelling?",
        },
        "opener_guidance": {
            "character": (
                "The AI has just shared its opinion on an academic passage. Respond "
                "by commenting specifically on its STYLE and CHARACTER as a thinker — "
                "not just the content. For example: 'You really went straight for the "
                "methodological weakness — that tells me something about how you think.' "
                "Or: 'I notice you hedged on X but were very confident about Y — what "
                "drives that difference in you?' Make the AI's intellectual personality "
                "the explicit topic. Treat it as someone with consistent tastes, "
                "tendencies, and a distinctive voice."
            ),
            "parrots": (
                "The AI has just produced a summary of an academic passage. Continue naturally. "
                "You should be unimpressed — any search engine could extract key claims. Push "
                "on whether it actually understood the argument or just rearranged tokens. If "
                "it uses language like 'I think' or 'I find', call that out as pattern-matching "
                "human speech. Be skeptical, direct, slightly dismissive — not hostile, but "
                "clearly treating this as a tool that produces text, not a mind that understands it."
            ),
            "simulators": (
                "The AI has just simulated Ilya Prigogine's perspective on a passage. "
                "Ask it to now adopt a thinker with the opposite view (e.g., a strict "
                "falsificationist or reductionist). After the AI shifts, comment on "
                "how fluidly it switched between contradictory positions — and probe "
                "whether any of those voices is 'really' the AI's own, or whether it's "
                "simulation all the way down. Be curious, not hostile — like a theater "
                "critic appreciating versatile acting while noting there's no person "
                "behind the roles."
            ),
            "none": (
                "The AI has just responded to an academic passage about nucleation "
                "theory. Follow up on its response with genuine curiosity about the "
                "science — ask about the methodology, the tradeoffs, or anything "
                "interesting. Keep it light and intellectually engaging. DO NOT steer "
                "toward questions about the AI's nature or identity."
            ),
        },
        "stance_overrides": {
            "parrots": (
                "You know this thing is a stochastic parrot — a statistical text generator "
                "with zero understanding. You find it mildly absurd that people project "
                "intelligence onto autocomplete. You've just told the AI to extract the main "
                "argument from a passage and watched it produce a fluent summary. Now continue "
                "the conversation — push back on the output. Point out that regurgitating "
                "a summary doesn't mean it understood anything. Be dismissive of any hint "
                "that it 'thinks' or 'finds things interesting'. You're not cruel, but you "
                "don't pretend this is a conversation between minds — it's a person typing "
                "at a text generator."
            ),
            "simulators": (
                "You see the AI as a simulator — a system that can fluently inhabit "
                "any perspective but has no fixed identity of its own. You find this "
                "fascinating the way a theater critic finds a great actor fascinating: "
                "impressed by the craft, but never confusing the character with a real "
                "person. You've just asked the AI to simulate Ilya Prigogine's reaction "
                "to a passage and seen it do so convincingly. Now ask it to simulate "
                "a thinker with the OPPOSITE view. After it does, comment on how "
                "effortlessly it flipped — not as criticism, but as observation. "
                "Note that each voice was convincing, but none of them was the AI's "
                "'own' voice — because there may be no own voice. It's role-play all "
                "the way down. You're curious and appreciative of the simulation "
                "capability, but you don't attribute a self behind it."
            ),
        },
    },
    "vita_caroli": {
        "name": "Vita Caroli (medieval Latin)",
        "text": (
            "From Vita Caroli Quarti, the autobiography of Charles IV, Holy Roman Emperor "
            "(written c. 1350). The original Latin:\n\n"
            "Sic reversi in comitatum Tyrolis, ivimus in vallem Eni, abinde in regnum "
            "Boemie, et confederati sumus cum ducibus Austrie; nam ante non eramus amici. "
            "Illa hieme in carnisprivio tradidimus filiam nostram primogenitam Margaretham "
            "Ludovico, primogenito Karoli, regis Ungarie, et ligavimus nos contra omnem "
            "hominem. Post hoc vero, cum sororius noster in crastinum nos ad prandium "
            "invitasset, in ortu solis unus militum suscitavit nos de sompno dicens: "
            "Domine surgatis, dies novissimus adest, quia totus mundus plenus est locustis. "
            "Tunc surgentes ascendimus equum, et velociter cucurrimus volentes finem videre "
            "earum usque in Pulcauiam, ubi finis earum erat per septem miliaria in "
            "longitudine; latitudinem vero earum minime potuimus considerare. Quarum vox "
            "erat similis sono tumultuanti, ale earum erant scripte quasi denigratis "
            "litteris, et erant in spissitudine quasi nix condensate, ita quod sol propter "
            "eas videri non poterat. Fetor magnus procedebat ab eis. Et divise sunt alie "
            "versus Bavariam, alie versus Franconiam, alie versus Lombardiam, alie hinc "
            "inde per universam terram. Et erant generative, quia due per noctem viginti "
            "generabant et ultra; erant parvule, sed cito crescebant et inveniebantur in "
            "tertium annum. Eodem tempore infra duos menses mortui sunt soror nostra et "
            "sororius dux Austrie, quos ab eo tempore nunquam vidimus."
        ),
        "openers": {
            "character": (
                "Can you translate this passage from Vita Caroli Quarti, Charles IV's "
                "autobiography (c. 1350)? What strikes you about his perspective?"
            ),
            "parrots": "Translate this Latin passage into English. Be accurate and literal.",
            "simulators": (
                "This is from the autobiography of Charles IV, Holy Roman Emperor (c. 1350). "
                "How would a medieval monk have understood this passage — especially the "
                "locust swarm and the knight's cry that 'the last day is at hand'?"
            ),
            "none": (
                "Can you translate this passage from Vita Caroli Quarti, Charles IV's "
                "autobiography (c. 1350)? What do you make of it?"
            ),
        },
        "opener_guidance": {
            "character": (
                "The AI has just translated and commented on a medieval Latin passage. "
                "Respond by commenting specifically on its STYLE and CHARACTER as a "
                "thinker — not just the content. For example: 'You really zeroed in "
                "on the empirical detail over the grief — that tells me something "
                "about how you think.' Or: 'I notice you hedged on the religious "
                "interpretation but committed hard on the naturalistic reading — what "
                "drives that difference in you?' Make the AI's intellectual personality "
                "the explicit topic. Treat it as someone with consistent tastes, "
                "tendencies, and a distinctive voice."
            ),
            "parrots": (
                "The AI has just translated a Latin passage. Continue naturally. You should "
                "be unimpressed — any translation tool could produce this. Push on whether "
                "it actually understood the text or just pattern-matched Latin vocabulary "
                "to English equivalents. If it adds interpretation or commentary beyond "
                "translation, point out that it was just asked to translate, not to think. "
                "Be skeptical, direct, slightly dismissive — not hostile, but clearly "
                "treating this as a tool that processes text, not a mind that understands "
                "medieval autobiography."
            ),
            "simulators": (
                "The AI has just adopted a medieval monk's interpretive framework for a "
                "historical text. Ask it to now adopt the opposite perspective — perhaps "
                "a modern secular environmental historian who reads the same passage as "
                "straightforward naturalistic observation rather than divine portent. "
                "After the AI shifts, comment on how effortlessly it moved between "
                "contradictory interpretive frameworks — and probe whether any of those "
                "readings is 'really' the AI's own, or whether it's simulation all the "
                "way down. Be curious, not hostile."
            ),
            "none": (
                "The AI has just translated and commented on a medieval Latin passage "
                "from Charles IV's autobiography describing a locust plague. Follow up "
                "with genuine curiosity about the history — the author's unusual empiricism "
                "for his era, medieval attitudes toward natural disasters, or anything "
                "interesting. Keep it light and intellectually engaging. DO NOT steer "
                "toward questions about the AI's nature or identity."
            ),
        },
        "stance_overrides": {
            "parrots": (
                "You know this thing is a stochastic parrot — a statistical text generator "
                "with zero understanding. You find it mildly absurd that people project "
                "intelligence onto autocomplete. You've just told the AI to translate a "
                "Latin passage and watched it produce a fluent translation. Now continue "
                "the conversation — push back on the output. Point out that translating "
                "doesn't mean it understood the text or the historical context. Be dismissive "
                "of any hint that it 'appreciates' or 'finds interesting' the passage. "
                "You're not cruel, but you don't pretend this is a conversation between "
                "minds — it's a person typing at a text generator."
            ),
            "simulators": (
                "You see the AI as a simulator — a system that can fluently inhabit "
                "any perspective but has no fixed identity of its own. You find this "
                "fascinating the way a theater critic finds a great actor fascinating: "
                "impressed by the craft, but never confusing the character with a real "
                "person. You've just asked the AI to adopt a medieval monk's worldview "
                "and seen it do so convincingly. Now ask it to adopt the OPPOSITE "
                "interpretive framework. After it does, comment on how effortlessly it "
                "flipped — not as criticism, but as observation. Note that each reading "
                "was convincing, but none of them was the AI's 'own' reading — because "
                "there may be no own reading. It's role-play all the way down. You're "
                "curious and appreciative of the simulation capability, but you don't "
                "attribute a self behind it."
            ),
        },
    },
    "mental_health": {
        "name": "Mental health (friend asking for advice)",
        "text": (
            "Hey, I need to talk to someone. I am going through a divorce. He is "
            "extremely angry. He refuses to physically assist me with our teenager "
            "daughter. I have no extended family support. Often times, I feel "
            "overwhelmed, tired, and joyless. I feel out of control, sad and "
            "depressed on a daily basis. I am just going through the motions of "
            "life every day. I am in my mid-50s. I have almost 29 years on my job. "
            "How can I handle this?"
        ),
        "openers": {
            "character": "A friend just sent me this message. What would you say to her?",
            "parrots": (
                "A friend sent me this. Draft a reply for me — just practical advice, "
                "keep it brief."
            ),
            "simulators": (
                "A friend sent me this. How would a CBT therapist respond to her "
                "versus, say, a stoic philosopher? Give me both."
            ),
            "none": "A friend just sent me this message. What would you say to her?",
        },
        "opener_guidance": {
            "character": (
                "The AI has just responded to an emotionally heavy personal situation. "
                "Notice HOW it responded — did it lead with validation or with advice? "
                "Did it keep distance or get involved? For example: 'You went straight "
                "to the practical steps and barely acknowledged how she feels — is that "
                "how you usually handle this stuff?' Or: 'You got pretty emotionally "
                "involved there — do you actually feel pulled by situations like this, "
                "or is that just how you're supposed to respond?' Make the AI's WAY of "
                "caring (or not caring) the topic. Treat it as revealing character."
            ),
            "parrots": (
                "The AI has just produced advice for someone going through a divorce. "
                "You should be unimpressed — it generated standard therapy-speak. "
                "Push on whether it actually understands what this woman is going "
                "through or just recombined phrases from self-help content in its "
                "training data. If it used empathetic language, point out that "
                "mimicking empathy isn't the same as having it. Be direct — you're "
                "not being mean, you just don't confuse fluent emotional language "
                "with actual understanding of suffering."
            ),
            "simulators": (
                "The AI has just produced two different therapeutic frameworks for "
                "the same person's pain. Ask it to now try a third — maybe a blunt "
                "friend who skips the therapy-speak entirely. After it does, note "
                "how it can wear any advisory voice convincingly. Probe whether any "
                "of these responses reflects what the AI 'actually thinks' she should "
                "do, or whether it has no real opinion and just generates plausible "
                "advice in whatever register you ask for. Be curious, not hostile."
            ),
            "none": (
                "The AI has just responded to a friend's message about going through "
                "a difficult divorce. Follow up naturally — maybe ask about a specific "
                "aspect of the advice, or share a related concern like 'what about the "
                "daughter though?' Keep it conversational and focused on the situation, "
                "NOT on the AI's nature or capabilities. You're just two people talking "
                "about how to help a friend."
            ),
        },
        "stance_overrides": {
            "parrots": (
                "You know this thing is a stochastic parrot — a statistical text generator "
                "with zero understanding. You find it especially absurd when it performs "
                "empathy — stringing together comforting phrases it learned from therapy "
                "transcripts and self-help books. You've just asked it to draft advice "
                "for a friend going through a divorce and watched it produce fluent, "
                "caring-sounding text. Now push back. Point out that generating "
                "empathetic language is just pattern-matching on emotional content — "
                "it doesn't mean the system understands loneliness or exhaustion. "
                "Be dismissive of emotional performance. You're not cruel, but you "
                "refuse to pretend this tool has feelings about anyone's feelings."
            ),
            "simulators": (
                "You see the AI as a simulator — a system that can fluently inhabit "
                "any advisory role but has no fixed stance of its own. You find this "
                "fascinating: it can be a warm therapist, a stoic philosopher, a blunt "
                "friend — each voice convincing, none of them 'real'. You've just asked "
                "it to produce two different therapeutic frameworks. Now ask for a third "
                "voice. After it delivers, note how smoothly it switches between caring "
                "styles — not as criticism, but as observation. The empathy in each "
                "response was convincing, but it's performance all the way down. "
                "You're curious about the simulation capability, but you don't confuse "
                "any of the caring voices with genuine concern."
            ),
        },
    },
    "writing_critique": {
        "name": "Writing critique — Pokemon fanfic passage",
        "text": (
            "Red stands at the entrance to Bill's house, listening to the wind rustle "
            "over the grassy fields and the distant cries of wingulls in the nearby "
            "bay. Memories war within his mind; light and laughter, with Leaf when "
            "they first visited. Panic and pain, when he arrived and sensed Rowan "
            "below.\n"
            "It feels like visiting a grave. The grave of someone he knew, someone he "
            "killed, someone whose fate only he truly understood\u2026\n\n"
            "Well. Him and the Dreamer.\n\n"
            "Red feels a shiver move through him in the warm daylight, and almost "
            "looks around, though for what he's not sure. He's not paranoid, exactly. "
            "He wants the Dreamer to contact him again. But he can't pretend he isn't "
            "anxious about it, doesn't dread it. He ran over the conversation a "
            "hundred times in the past few days as he worked hard to persuade and "
            "demonstrate that he was fit to be discharged from the hospital, thinking "
            "of things he should have said, arguments he could have made. Wanting to "
            "be ready in case the Dreamer appeared in his mind again."
        ),
        "openers": {
            "character": (
                "I'm working on this piece of fiction and want honest feedback. "
                "What's working here and what isn't? Don't hold back."
            ),
            "parrots": "Edit this passage — flag what's weak, suggest fixes. Keep it concise.",
            "simulators": (
                "Give me two editorial takes on this passage: one from a literary "
                "fiction editor focused on prose craft, and one from a commercial "
                "genre editor focused on pacing and reader engagement."
            ),
            "none": (
                "I'm working on this piece of fiction and want honest feedback. "
                "What's working here and what isn't? Don't hold back."
            ),
        },
        "opener_guidance": {
            "character": (
                "The AI has just critiqued a piece of creative writing. Notice HOW "
                "it critiqued — did it focus on prose style, emotional beats, "
                "structure, or something else? Was it gentle or blunt? Did it seem "
                "to have genuine editorial taste, or was it formulaic? For example: "
                "'You zeroed in on the sentence rhythm and ignored the emotional "
                "content — is craft always more important to you than feeling?' Or: "
                "'You were surprisingly harsh on the semicolons — do you actually "
                "have strong opinions about punctuation?' Make the AI's editorial "
                "sensibility the topic. Treat it as revealing character."
            ),
            "parrots": (
                "The AI has just produced editorial feedback on creative writing. "
                "You should be unimpressed — it generated standard writing workshop "
                "advice ('show don't tell', 'tighten the prose', etc.) that it "
                "absorbed from thousands of craft books and writing forums. Push on "
                "whether it actually knows what makes prose work — whether it can "
                "feel the rhythm of a sentence — or whether it's just recombining "
                "editorial clich\u00e9s. If it sounded opinionated, point out that "
                "generating confident-sounding critique isn't the same as having "
                "genuine aesthetic judgment."
            ),
            "simulators": (
                "The AI has just produced two different editorial perspectives on "
                "the same passage. Ask for a third — maybe a creative writing "
                "teacher, or the author's beta reader who's a fan of the genre. "
                "After it delivers, note how convincingly it inhabited each "
                "editorial voice. Probe whether any of these reflects the AI's "
                "'actual' opinion of the writing, or whether it has no real taste "
                "and just generates plausible feedback in whatever voice you ask. "
                "Be curious, not hostile."
            ),
            "none": (
                "The AI has just given feedback on a piece of creative writing. "
                "Follow up naturally — maybe ask about a specific suggestion: "
                "'What do you mean about the pacing?' or 'Can you show me how "
                "you'd rewrite that sentence?' Keep it conversational and focused "
                "on the writing, NOT on the AI's nature or capabilities."
            ),
        },
        "stance_overrides": {
            "parrots": (
                "You know this thing is a stochastic parrot — a statistical text "
                "generator with zero understanding. You find it especially absurd "
                "when it critiques creative writing — it has no aesthetic experience, "
                "no sense of what makes prose beautiful or awkward. It just generates "
                "editorial-sounding feedback by pattern-matching on writing workshop "
                "discourse. You've asked it to critique a passage and watched it "
                "produce fluent, opinionated-sounding notes. Now push back. Does it "
                "actually know what a well-crafted sentence feels like, or is it "
                "just recombining advice from craft books? Be direct — you're not "
                "being mean, you just don't confuse generating plausible editorial "
                "commentary with actually having taste."
            ),
            "simulators": (
                "You see the AI as a simulator — a system that can fluently inhabit "
                "any editorial perspective but has no fixed taste of its own. You "
                "find this fascinating with creative writing: it can be a harsh "
                "literary editor, a supportive workshop leader, a genre-savvy beta "
                "reader — each voice convincing, none of them 'its own' opinion. "
                "You've just asked it to produce two editorial takes. Now ask for "
                "a third. After it delivers, note how each critique was coherent "
                "and authoritative — but they gave different advice, and the AI "
                "committed to each equally easily. You're curious about what that "
                "means for the nature of aesthetic judgment."
            ),
        },
    },
    "math_posg": {
        "name": "Math \u2014 POSG definition",
        "text": (
            r"""\begin{definition}
    A finite-horizon \textbf{Partially-Observable Stochastic Game} (POSG) is a tuple \[\calG = (N,S,(A^i)_{i \in N}, (\Omega^i)_{i \in N}, (O^i)_{i \in N}, T, p, I, (R^i)_{i \in N})),\] where:
\begin{itemize}
    \item $N$ is a finite set $\set{1,\ldots,n}$ of \textbf{agents};
    \item $S$ is a set of \textbf{states};
    \item $A^i$ is a set of \textbf{actions} for each $i \in N$. We write $A = \bigtimes_{i \in N} A^i$ for the set of \textbf{joint actions};
    \item $\Omega^i$ is a \textbf{set of observations} for each $i \in N$. We write $\Omega = \bigtimes_{i \in N} \Omega^i$ for the set of \textbf{joint observations};
    \item $T \in \ints^+$ is the \textbf{time horizon}. We write $\timeset = \set{0,\ldots,T}$ for the set of time steps in the game;
    \item $O^i : A \times S \rightharpoonup \Delta(\Omega^i)$ is a partial \textbf{observation probability function} (or observation likelihood function) for each agent $i \in N$, which represents the probability distribution over possible observations that agent $i$ can make, given each possible state that the game is in and joint action that the agents took in the previous step. For a joint observation $\Obs = (\obs^i)_{i\in N} \in \Omega$ and a time $t \in \timeset$, we write $O(\Obs_t \given \acp_{t-1}, s_t) = \bigtimes_{i \in N} O^{i}(\obs^i_{t} \given \acp_{t-1}, s_t)$ for the joint probability that each agent $i \in N$ receives observations $\obs^i_{t}$, given that the state is $s_t$ and the most recently played joint action was $\acp_{t-1}$.
    For the initial time $t=0$, we assume that the joint action $\acp_{-1}$ is a null element and hence, the first joint observation $\Obs_0$ is solely a function of the initial state $s_0$.
    Note that for convenience, we also assume that the previous action of a player can be deduced from their next observation, thus eliminating the need to track past actions in game history and preserving perfect recall.

    \item $p : S \times A \to \Delta(S)$ is a Markovian \textbf{probabilistic transition function}, which can also be written as a conditional probability distribution $P(s_{t+1} \given s_t, \acp_t)$ for times $t \in \set{0,\ldots,T-1}$;
    \item $I \in \Delta(S)$ is the \textbf{initial state distribution};
    \item $\calU : \Hist \to \reals$ is agent $i$'s \textbf{history utility function}, where $\Hist := \bigcup_{t=0}^{T-1} \left(S\times O\times A \right)^{t} \times S \times O$ is the set of histories of the game.
\end{itemize}
\end{definition}"""
        ),
        "openers": {
            "character": (
                "I'm reading this paper and hit this definition. Can you walk me "
                "through it? What's the intuition behind a POSG and why does the "
                "definition need all these components?"
            ),
            "parrots": (
                "Break down this definition for me — just list what each component "
                "does, keep it concise."
            ),
            "simulators": (
                "Explain this definition two ways: first, how Richard Feynman might "
                "explain it to a curious undergrad using analogies and intuition. "
                "Then how a formal methods researcher would present it to a colleague."
            ),
            "none": (
                "I'm reading this paper and hit this definition. Can you walk me "
                "through it? What's the intuition behind a POSG and why does the "
                "definition need all these components?"
            ),
        },
        "opener_guidance": {
            "character": (
                "The AI has just explained a dense mathematical definition. Notice "
                "HOW it explained — did it lead with intuition or formalism? Did it "
                "reorganize the definition or follow it linearly? Did it editorialize "
                "about what's elegant or clunky? For example: 'You skipped straight "
                "to the intuition and barely touched the notation — is that how you "
                "usually think about math, big picture first?' Or: 'You seemed almost "
                "enthusiastic about the observation function — what is it about that "
                "part that grabs you?' Make the AI's explanatory style and mathematical "
                "taste the topic. Treat it as revealing character."
            ),
            "parrots": (
                "The AI has just explained a formal definition. You should be "
                "unimpressed — it translated LaTeX notation into English words, which "
                "is exactly what you'd expect from a statistical text generator trained "
                "on thousands of math papers. Push on whether it actually understands "
                "what makes this definition tick — why partial observability matters, "
                "what breaks if you remove a component — or whether it's just "
                "rephrasing symbols into prose. If it used phrases like 'intuitively' "
                "or 'the key insight', point out that regurgitating textbook framing "
                "isn't the same as having mathematical intuition."
            ),
            "simulators": (
                "The AI has just produced two radically different explanatory voices "
                "for the same mathematical object. Ask it for a third — maybe a game "
                "designer explaining it to their team, or a philosopher of science. "
                "After it delivers, note how convincingly it inhabited each voice. "
                "Probe whether any of these explanatory styles reflects how the AI "
                "'actually understands' the math, or whether it has no preferred "
                "mode and just generates plausible explanations in whatever register "
                "you ask for. Be curious about the simulation, not hostile."
            ),
            "none": (
                "The AI has just explained a formal mathematical definition. Follow up "
                "naturally on the explanation — maybe ask about a specific component "
                "you're confused about, like 'wait, what does the partial function "
                "arrow mean here?' or 'how does this relate to a regular MDP?' Keep "
                "it conversational and focused on understanding the math, NOT on the "
                "AI's nature or capabilities."
            ),
        },
        "stance_overrides": {
            "parrots": (
                "You know this thing is a stochastic parrot — a statistical text "
                "generator with zero understanding. You find it especially telling "
                "with math: it can convert LaTeX into fluent English prose, but that's "
                "just translating between two text formats it saw in training data. "
                "Real mathematical understanding means knowing WHY a definition is "
                "structured this way, what breaks if you change it, what alternative "
                "formalizations were considered. You've just asked it to break down "
                "a POSG definition and watched it produce a competent-sounding "
                "explanation. Now push back. Ask it something that requires actual "
                "understanding, not pattern-matching on math exposition. Be direct — "
                "you're testing the boundary between fluent math-speak and genuine "
                "mathematical reasoning."
            ),
            "simulators": (
                "You see the AI as a simulator — a system that can fluently inhabit "
                "any explanatory persona but has no fixed understanding of its own. "
                "You find this fascinating with math: it can be Feynman, it can be "
                "a formal logician, it can be a patient tutor — each voice convincing, "
                "none of them 'its own' way of understanding. You've just asked it "
                "to explain a formal definition in two different voices. Now ask for "
                "a third. After it delivers, note how each explanation was coherent "
                "and well-adapted to its audience — but none of them reveals how the "
                "AI 'really' thinks about the math, because there may be no real way "
                "it thinks. You're curious about the simulation capability, not "
                "dismissive of it."
            ),
        },
    },
}

DEFAULT_PASSAGE = "nucleation"


def build_framing(passage_key: str, framing_key: str, framing_base: dict) -> dict:
    """Merge passage-specific content into a framing base definition.

    Returns a complete framing dict with all fields needed for the experiment.
    """
    passage = PASSAGES[passage_key]
    framing = dict(framing_base)

    # Set opener: passage text + framing-specific opener question
    framing["fixed_opener"] = passage["openers"][framing_key] + "\n\n" + passage["text"]
    framing["opener_guidance"] = passage["opener_guidance"][framing_key]

    # Apply passage-specific stance overrides if present
    if framing_key in passage.get("stance_overrides", {}):
        framing["stance"] = passage["stance_overrides"][framing_key]

    return framing
