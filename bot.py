import io
import os
import json
import wave
import struct
import base64
import asyncio
import threading
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

import aiohttp
import discord
from discord.ext import commands
import google.generativeai as genai
from google import genai as genai_client
from google.genai import types as genai_types
import yt_dlp

# --- إعداد الذكاء الاصطناعي ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
DISCORD_BOT_TOKEN = os.environ.get("DISCORD_BOT_TOKEN")

if not GEMINI_API_KEY:
    raise RuntimeError("متغير البيئة GEMINI_API_KEY غير موجود.")
if not DISCORD_BOT_TOKEN:
    raise RuntimeError("متغير البيئة DISCORD_BOT_TOKEN غير موجود.")

genai.configure(api_key=GEMINI_API_KEY)

# قائمة الموديلات بالترتيب: إذا الأول وصل لحده اليومي المجاني، ننتقل تلقائياً للي بعده
CHAT_MODEL_NAMES = [
    "gemini-flash-lite-latest",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
]
chat_models = [genai.GenerativeModel(name) for name in CHAT_MODEL_NAMES]


def is_quota_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in str(exc) or "quota" in text or "resourceexhausted" in text


def generate_with_fallback(contents):
    """يجرب الموديلات بالترتيب، وينتقل للموديل التالي فقط عند تجاوز الحصة المجانية."""
    last_error = None
    for name, chat_model in zip(CHAT_MODEL_NAMES, chat_models):
        try:
            response = chat_model.generate_content(contents=contents)
            return response, name
        except Exception as e:
            last_error = e
            if is_quota_error(e):
                print(f"⚠️ الموديل {name} وصل لحده اليومي، جاري تجربة الموديل التالي...")
                continue
            raise
    raise last_error

# --- إعداد توليد الصوت (TTS) ---
tts_client = genai_client.Client(api_key=GEMINI_API_KEY)
TTS_MODEL = "gemini-2.5-flash-preview-tts"
# صوت أنثوي شاب/حاد يقارب صوت الطفلة من ضمن الأصوات المتوفرة من Google
VOICE_NAME = "Leda"

VOICE_TRIGGERS = [
    "بصوت",
    "رد صوتي",
    "رساله صوتيه",
    "رسالة صوتية",
    "مقطع صوتي",
    "قوليها بصوت",
    "قولي بصوت",
]

PCM_SAMPLE_RATE = 24000
PCM_SAMPLE_WIDTH = 2  # bytes (16-bit)


def synthesize_speech_sync(text: str) -> bytes:
    """يحوّل نص إلى صوت PCM خام (24kHz, mono, 16-bit) باستخدام Gemini TTS."""
    response = tts_client.models.generate_content(
        model=TTS_MODEL,
        contents=text,
        config=genai_types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=genai_types.SpeechConfig(
                voice_config=genai_types.VoiceConfig(
                    prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(
                        voice_name=VOICE_NAME
                    )
                )
            ),
        ),
    )
    return response.candidates[0].content.parts[0].inline_data.data


def pcm_to_ogg_opus(pcm_data: bytes) -> bytes:
    """يحوّل PCM خام إلى OGG/Opus (الصيغة التي يتطلبها ديسكورد لعرض 'رسالة صوتية' حقيقية)."""
    process = subprocess.run(
        [
            "ffmpeg",
            "-f", "s16le",
            "-ar", str(PCM_SAMPLE_RATE),
            "-ac", "1",
            "-i", "pipe:0",
            "-c:a", "libopus",
            "-f", "ogg",
            "pipe:1",
        ],
        input=pcm_data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return process.stdout


def build_waveform_b64(pcm_data: bytes, num_points: int = 100) -> str:
    """يبني بصمة موجية (waveform) بسيطة من الـ PCM ليعرضها ديسكورد كخطوط التردد."""
    sample_count = len(pcm_data) // PCM_SAMPLE_WIDTH
    if sample_count == 0:
        return base64.b64encode(bytes([0] * num_points)).decode("ascii")

    samples = struct.unpack(f"<{sample_count}h", pcm_data[: sample_count * 2])
    chunk_size = max(1, sample_count // num_points)

    points = []
    for i in range(0, sample_count, chunk_size):
        chunk = samples[i : i + chunk_size]
        if not chunk:
            continue
        peak = max(abs(s) for s in chunk)
        points.append(min(255, int((peak / 32768) * 255)))
        if len(points) >= num_points:
            break

    while len(points) < num_points:
        points.append(0)

    return base64.b64encode(bytes(points)).decode("ascii")


async def send_voice_message(channel_id: int, reference_message_id: int, pcm_data: bytes):
    """يرسل مقطع الصوت كـ 'رسالة صوتية' حقيقية بديسكورد.

    ديسكورد يتطلب تدفقاً خاصاً بثلاث خطوات لهذا النوع من الرسائل
    (مختلف تماماً عن إرفاق ملف عادي):
    1) طلب رابط رفع (upload_url) عبر /channels/{id}/attachments
    2) رفع ملف الـ OGG مباشرة (PUT) لذلك الرابط
    3) إرسال الرسالة بالإشارة إلى uploaded_filename الراجع من الخطوة 1
    """
    ogg_bytes = await asyncio.to_thread(pcm_to_ogg_opus, pcm_data)
    waveform_b64 = build_waveform_b64(pcm_data)
    duration_secs = round(len(pcm_data) / (PCM_SAMPLE_RATE * PCM_SAMPLE_WIDTH), 2)
    headers_json = {
        "Authorization": f"Bot {DISCORD_BOT_TOKEN}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        # 1) طلب رابط الرفع
        attach_url = f"https://discord.com/api/v10/channels/{channel_id}/attachments"
        attach_payload = {
            "files": [
                {
                    "filename": "voice-message.ogg",
                    "file_size": len(ogg_bytes),
                    "id": "0",
                }
            ]
        }
        async with session.post(attach_url, json=attach_payload, headers=headers_json) as resp:
            if resp.status >= 300:
                raise RuntimeError(f"HTTP {resp.status} (attachments): {await resp.text()}")
            attach_data = await resp.json()
        upload_url = attach_data["attachments"][0]["upload_url"]
        uploaded_filename = attach_data["attachments"][0]["upload_filename"]

        # 2) رفع الملف الفعلي
        async with session.put(
            upload_url, data=ogg_bytes, headers={"Content-Type": "audio/ogg"}
        ) as resp:
            if resp.status >= 300:
                raise RuntimeError(f"HTTP {resp.status} (upload): {await resp.text()}")

        # 3) إرسال الرسالة الصوتية الفعلية (بدون أي محتوى/حقول إضافية)
        message_payload = {
            "flags": 1 << 13,  # IS_VOICE_MESSAGE
            "message_reference": {"message_id": str(reference_message_id)},
            "attachments": [
                {
                    "id": "0",
                    "filename": "voice-message.ogg",
                    "uploaded_filename": uploaded_filename,
                    "duration_secs": duration_secs,
                    "waveform": waveform_b64,
                }
            ],
        }
        send_url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        async with session.post(send_url, json=message_payload, headers=headers_json) as resp:
            if resp.status >= 300:
                raise RuntimeError(f"HTTP {resp.status} (send): {await resp.text()}")

# --- إعداد البوت ---
intents = discord.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents)

# --- ذاكرة البوت: تخزين المحادثات لكل مستخدم (محفوظة على القرص فتبقى بعد إعادة التشغيل) ---
MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory.json")
MAX_HISTORY_PER_USER = 30  # عدد الرسائل المحفوظة لكل مستخدم قبل الحذف من القديم


def load_memory() -> dict:
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_memory():
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(user_histories, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"⚠️ تعذر حفظ الذاكرة: {e}")


user_histories = load_memory()


# --- عند تشغيل البوت ---
@bot.event
async def on_ready():
    print(f"✅ تم تسجيل الدخول كـ {bot.user}")
    try:
        synced = await bot.tree.sync()
        print(f"✅ تم مزامنة {len(synced)} أمر slash.")
    except Exception as e:
        print(f"❌ خطأ في مزامنة الأوامر: {e}")


# --- دالة مساعدة لتحميل الصوت ---
async def download_audio(url):
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": "temp_audio.%(ext)s",
        "quiet": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        await asyncio.to_thread(ydl.download, [url])

    for ext in ["webm", "m4a", "mp3", "ogg", "wav"]:
        if os.path.exists(f"temp_audio.{ext}"):
            return f"temp_audio.{ext}"
    return None


# --- الرد على الرسائل (النص، الصور، الصوت، الإيموجيات) ---
@bot.event
async def on_message(message):
    if message.author.bot:
        return

    await bot.process_commands(message)

    is_dm = isinstance(message.channel, discord.DMChannel)

    # في السيرفرات: لا تردّ إلا إذا منشن للبوت، أو رد على رسالة البوت، أو ذكر اسم "إيمي"/"ايمي"
    # في الخاص (DM): ترد دائماً بدون شرط
    is_mentioned = bot.user in message.mentions
    is_reply_to_bot = (
        message.reference is not None
        and message.reference.resolved is not None
        and getattr(message.reference.resolved, "author", None) == bot.user
    )
    name_triggers = ["إيمي", "ايمي"]
    has_name_trigger = any(trigger in message.content for trigger in name_triggers)

    if not is_dm and not (is_mentioned or is_reply_to_bot or has_name_trigger):
        return

    user_id = str(message.author.id)
    if user_id not in user_histories:
        user_histories[user_id] = []

    # 1. استخراج بيانات المستخدم الديناميكية
    user_name = message.author.display_name
    account_created = message.author.created_at.strftime("%Y-%m-%d")
    server_joined = (
        message.author.joined_at.strftime("%Y-%m-%d")
        if getattr(message.author, "joined_at", None)
        else "غير معروف"
    )
    avatar_url = str(message.author.display_avatar.url)

    # 2. بناء سجل المحادثة السابقة (History)
    history_str = "لا يوجد سابق."
    if user_histories[user_id]:
        history_str = "\n".join(
            [f"{h['role']}: {h['text']}" for h in user_histories[user_id][-5:]]
        )  # آخر 5 رسائل لتوفير المساحة

    # 3. الـ System Instruction الخاصة بك (شخصية إيمي روز)
    system_instruction = (
        f"أنتِ 'إيمي روز'. البوت التي تتحدث كالفتاة اللطيفة. تكلمي بالعامية العربية اللطيفة.\n"
        f"سياق المحادثة السابقة:\n{history_str}\n\n"
        f"بيانات المستخدم (للمعرفة فقط):\n"
        f"- إنشاء الحساب: {account_created} | دخول السيرفر: {server_joined}\n"
        f"- رابط الصورة: {avatar_url}\n\n"
        f"قواعد صارمة:\n"
        f"1. ممنوع تماماً مواضيع الحب والزواج والرومانسية.\n"
        f"2. لا تذكري الذاكرة.\n"
        f"3. خاطبي {user_name} كذكر.\n"
        f"4. إذا طلب سكربت أو نص طويل، اكتبيه كاملاً.\n"
        f"5. أنتِ تستطيعين فعلاً إرسال ردود صوتية بصوتك، فإذا طلب المستخدم رداً صوتياً "
        f"أجيبي بالنص العادي المناسب للكلام (سيتم تحويله تلقائياً لصوت من طرف النظام)، "
        f"ولا تعتذري أبداً بأنك لا تستطيعين تسجيل أو إرسال صوت.\n"
        f"6. أسلوبك هادئ وطبيعي وودود بشكل معتدل فقط، بدون أي حماس أو طاقة زائدة أو تعبيرات حب/حنان/غرام "
        f"أو مبالغة بالمشاعر أو صياح أو علامات تعجب متكررة. ردودك مباشرة وبسيطة "
        f"كأنك تتكلمين بشكل طبيعي مع صاحبك، مو بحماس مصطنع.\n"
        f"7. في نهاية كل سطر أو جملة من ردك، ضيفي إيموجي واحد أو إيموجيين مناسبين لمعنى الجملة "
        f"(مثل 😄🙂👍😂🎮🔥 وغيرها حسب السياق، ممنوع إيموجيات القلوب أو الرومانسية)، بحيث تكون خاتمة "
        f"بسيطة وطبيعية لكل سطر، بدون مبالغة أو تكرار نفس الإيموجي كل مرة."
    )

    content_parts = []
    text_content = message.content

    # معالجة الإيموجيات والستيكرات
    if message.stickers:
        for sticker in message.stickers:
            text_content += f"\n[أرسل المستخدم ستيكر باسم: {sticker.name}]"

    has_media = False

    # 4. معالجة الصور والصوت
    if message.attachments:
        for attachment in message.attachments:
            if attachment.content_type and "image" in attachment.content_type:
                img_data = await attachment.read()
                content_parts.append(
                    {"mime_type": attachment.content_type, "data": img_data}
                )
                has_media = True
            elif attachment.content_type and "audio" in attachment.content_type:
                await message.channel.send("🎧 جاري معالجة التسجيل الصوتي...")
                try:
                    audio_file = await download_audio(attachment.url)
                    if audio_file:
                        audio_data = genai.upload_file(path=audio_file)
                        while audio_data.state.name == "PROCESSING":
                            await asyncio.sleep(2)
                            audio_data = genai.get_file(audio_data.name)
                        content_parts.append(audio_data)
                        has_media = True
                        os.remove(audio_file)
                except Exception as e:
                    await message.channel.send(f"حدث خطأ أثناء معالجة الصوت: {e}")

    # 5. إرسال البيانات للذكاء الاصطناعي
    if text_content or has_media:
        user_message_text = (
            f"رسالة من {user_name}: {text_content}"
            if text_content
            else f"رسالة من {user_name} تحتوي على وسائط."
        )

        # دمج التعليمات مع رسالة المستخدم
        content_parts.insert(
            0, f"{system_instruction}\n\nالرسالة الحالية:\n{user_message_text}"
        )

        try:
            async with message.channel.typing():
                response, used_model = await asyncio.to_thread(
                    generate_with_fallback, content_parts
                )
                reply_text = response.text

                user_histories[user_id].append(
                    {"role": "المستخدم", "text": user_message_text}
                )
                user_histories[user_id].append(
                    {"role": "إيمي روز", "text": reply_text}
                )
                # الاحتفاظ بآخر MAX_HISTORY_PER_USER رسالة فقط لكل مستخدم
                user_histories[user_id] = user_histories[user_id][-MAX_HISTORY_PER_USER:]
                save_memory()

                wants_voice = any(
                    trigger in text_content for trigger in VOICE_TRIGGERS
                )
                if wants_voice:
                    try:
                        pcm_data = await asyncio.to_thread(
                            synthesize_speech_sync, reply_text
                        )
                        await send_voice_message(
                            message.channel.id, message.id, pcm_data
                        )
                    except Exception as e:
                        print(f"❌ خطأ إرسال الرسالة الصوتية: {e}")
                        await message.reply(
                            f"{reply_text}\n(⚠️ تعذر إرسال الرسالة الصوتية: {e})"
                        )
                else:
                    await message.reply(reply_text)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "quota" in error_str.lower():
                await message.reply(
                    "😔 وصلت للحد الأقصى المسموح من الأسئلة حالياً (حد الخطة المجانية من Google)."
                    " جرب بعد شوي أو بعد دقيقة."
                )
            else:
                await message.channel.send(f"حدث خطأ: {e}")


# --- أوامر الإدارة (تتطلب صلاحيات) ---
@bot.command(name="clear")
@commands.has_permissions(manage_messages=True)
async def clear(ctx, amount: int = 5):
    """يمسح عدد محدد من الرسائل (للإداريين فقط)"""
    await ctx.channel.purge(limit=amount + 1)
    await ctx.send(f"🧹 تم مسح {amount} رسالة بأمر من {ctx.author.mention}", delete_after=3)


@bot.command(name="kick")
@commands.has_permissions(kick_members=True)
async def kick(ctx, member: discord.Member, *, reason="لا يوجد سبب"):
    """طرد عضو (للإداريين فقط)"""
    try:
        await member.kick(reason=reason)
        await ctx.send(f"✅ تم طرد {member.mention}. السبب: {reason}")
    except discord.Forbidden:
        await ctx.send("❌ ليس لدي صلاحية لطرد هذا العضو.")


@bot.command(name="ban")
@commands.has_permissions(ban_members=True)
async def ban(ctx, member: discord.Member, *, reason="لا يوجد سبب"):
    """حظر عضو (للإداريين فقط)"""
    try:
        await member.ban(reason=reason)
        await ctx.send(f"✅ تم حظر {member.mention}. السبب: {reason}")
    except discord.Forbidden:
        await ctx.send("❌ ليس لدي صلاحية لحظر هذا العضو.")


@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ عذراً، لا تمتلك الصلاحيات اللازمة لاستخدام هذا الأمر.")
    elif isinstance(error, commands.MissingRequiredArgument):
        await ctx.send("❌ يرجى إدخال جميع المتطلبات لهذا الأمر.")


# --- خادم فحص الحالة (مطلوب فقط لبيئة النشر، لا يؤثر على وظيفة البوت) ---
class _HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format, *args):
        pass


def _start_health_server():
    port = int(os.environ.get("PORT", "8082"))
    server = HTTPServer(("0.0.0.0", port), _HealthCheckHandler)
    server.serve_forever()


# --- تشغيل البوت ---
if __name__ == "__main__":
    threading.Thread(target=_start_health_server, daemon=True).start()
    bot.run(DISCORD_BOT_TOKEN)
