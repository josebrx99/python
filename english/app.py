import gradio as gr
import pandas as pd
import unicodedata
import time
import os
from gtts import gTTS
from pydub import AudioSegment

# ---------------------------
# Cargar datos
# ---------------------------
df = pd.read_excel("traducciones.xlsx")

if "correct_count" not in df.columns:
    df["correct_count"] = 0

if "wrong_count" not in df.columns:
    df["wrong_count"] = 0

# ordenar por frecuencia
df = df.sort_values(by="freq", ascending=False).reset_index(drop=True)

# ---------------------------
# Utils
# ---------------------------
def normalize(text):
    if text is None:
        return ""
    text = text.lower().strip()
    text = ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )
    return text

# ---------------------------
# Métricas
# ---------------------------
def get_metrics(df):
    learned = (df["correct_count"] >= 2).sum()
    learning = ((df["correct_count"] < 2) & (df["wrong_count"] > 0)).sum()
    return learned, learning

# ---------------------------
# Audio (gTTS + velocidad)
# ---------------------------
def generate_audio(word, slow=False):
    base_file = "audio_base.mp3"
    out_file = "audio_final.mp3"

    tts = gTTS(text=word, lang='en')
    tts.save(base_file)

    if slow:
        audio = AudioSegment.from_file(base_file)
        slower = audio._spawn(audio.raw_data, overrides={
            "frame_rate": int(audio.frame_rate * 0.75)
        }).set_frame_rate(audio.frame_rate)
        slower.export(out_file, format="mp3")
    else:
        os.replace(base_file, out_file)

    return out_file

# ---------------------------
# Palabras aleatorias
# ---------------------------
def get_random_words(df, exclude_word):
    sample_df = df[df["word"] != exclude_word]
    sample = sample_df.sample(min(3, len(sample_df)))

    html = "<div style='text-align:center; margin-top:15px;'>"
    html += "<div style='font-size:20px; font-weight:bold;'>🔀 Otras palabras</div><br>"

    for _, row in sample.iterrows():
        html += f"<div style='font-size:18px;'>{row['word']} → {row['translation']}</div>"

    html += "</div>"
    return html

# ---------------------------
# Estado
# ---------------------------
def init_state():
    return {
        "cooldown": {},   # palabra -> timestamp futuro
        "current": None
    }

# ---------------------------
# Selección inteligente
# ---------------------------
def get_next_word(state):
    now = time.time()
    candidates = []

    for _, row in df.iterrows():
        word = row["word"]

        if row["correct_count"] >= 2:
            continue

        available_time = state["cooldown"].get(word, 0)
        if now < available_time:
            continue

        prioridad = row["freq"] + (row["wrong_count"] * 50)
        candidates.append((prioridad, row))

    if not candidates:
        state["current"] = None
        return state

    candidates.sort(key=lambda x: x[0], reverse=True)
    state["current"] = candidates[0][1].to_dict()
    return state

# ---------------------------
# Formato palabra
# ---------------------------
def format_word(word):
    return f"""
    <div style='font-size:48px; font-weight:bold; text-align:center;'>
    {word}
    </div>
    """

# ---------------------------
# Evaluar respuesta (texto)
# ---------------------------
def submit_answer(user_input, state, slow_mode):

    if state["current"] is None:
        learned, learning = get_metrics(df)
        return "⏳ Esperando...", None, "", "", str(learned), str(learning), state

    word = state["current"]["word"]
    idx = df.index[df["word"] == word][0]

    correct_answer = normalize(state["current"]["translation"])
    user_answer = normalize(user_input)

    if user_answer == correct_answer:
        df.loc[idx, "correct_count"] += 1

        feedback = "<div style='font-size:28px; font-weight:bold; color:green; text-align:center;'>✅ Correcto</div>"
        state["cooldown"][word] = time.time() + 20
    else:
        df.loc[idx, "wrong_count"] += 1

        feedback = f"<div style='font-size:28px; font-weight:bold; color:red; text-align:center;'>❌ {state['current']['translation']}</div>"
        state["cooldown"][word] = time.time() + 5

    extra = get_random_words(df, word)
    full_feedback = feedback + extra

    state = get_next_word(state)

    learned, learning = get_metrics(df)

    if state["current"] is None:
        return "🎉 Terminaste", None, "", full_feedback, str(learned), str(learning), state

    next_word = state["current"]["word"]
    audio = generate_audio(next_word, slow_mode)

    return (
        format_word(next_word),
        audio,
        "",
        full_feedback,
        str(learned),
        str(learning),
        state
    )

# ---------------------------
# Evaluar voz 🎙️
# ---------------------------
def submit_voice(audio_input, state, slow_mode):
    # audio_input es ruta al archivo grabado
    if audio_input is None:
        return submit_answer("", state, slow_mode)

    # ⚠️ Aquí solo simulamos validación simple
    # (speech-to-text real requeriría API externa)
    return submit_answer("voz", state, slow_mode)

# ---------------------------
# Repetir audio 🔁
# ---------------------------
def repeat_audio(state, slow_mode):
    if state["current"] is None:
        return None
    return generate_audio(state["current"]["word"], slow_mode)

# ---------------------------
# Inicio
# ---------------------------
def start_app(state, slow_mode):
    state = get_next_word(state)
    learned, learning = get_metrics(df)

    if state["current"] is None:
        return "🎉 Todo aprendido", None, "", "", str(learned), str(learning), state

    word = state["current"]["word"]
    audio = generate_audio(word, slow_mode)

    return format_word(word), audio, "", "", str(learned), str(learning), state

# ---------------------------
# UI
# ---------------------------
with gr.Blocks() as app:

    state = gr.State(init_state())

    gr.Markdown("<h1 style='text-align:center;'>🧠 Aprende Inglés PRO</h1>")

    slow_mode = gr.Checkbox(label="🐢 Modo lento")

    with gr.Row():
        total = gr.Textbox(value=str(len(df)), label="📊 Total", interactive=False)
        learned = gr.Textbox(label="✅ Aprendidas", interactive=False)
        learning = gr.Textbox(label="🔥 En aprendizaje", interactive=False)

    word = gr.Markdown()
    audio = gr.Audio(autoplay=True)

    user_input = gr.Textbox(label="✍️ Escribe la traducción")
    mic = gr.Audio(source="microphone", type="filepath", label="🎙️ Habla")

    repeat_btn = gr.Button("🔁 Repetir audio")

    feedback = gr.Markdown()

    user_input.submit(
        submit_answer,
        inputs=[user_input, state, slow_mode],
        outputs=[word, audio, user_input, feedback, learned, learning, state]
    )

    mic.change(
        submit_voice,
        inputs=[mic, state, slow_mode],
        outputs=[word, audio, user_input, feedback, learned, learning, state]
    )

    repeat_btn.click(
        repeat_audio,
        inputs=[state, slow_mode],
        outputs=audio
    )

    app.load(
        start_app,
        inputs=[state, slow_mode],
        outputs=[word, audio, user_input, feedback, learned, learning, state]
    )

app.launch()