from __future__ import annotations

import asyncio
import logging
from dotenv import load_dotenv
from time import perf_counter

from livekit import rtc, api
from livekit.agents import (
    AutoSubscribe,
    JobContext,
    JobProcess,
    WorkerOptions,
    cli,
    llm,
)
from livekit.agents.multimodal import MultimodalAgent
from livekit.agents.pipeline import AgentCallContext, VoicePipelineAgent
from livekit.plugins import deepgram, openai, silero, turn_detector
from livekit.plugins import openai


# load environment variables, this is optional, only used for local development
load_dotenv(dotenv_path=".env.local")
logger = logging.getLogger("outbound-caller")
logger.setLevel(logging.INFO)

_default_instructions = (
    "You are a scheduling assistant for a dental practice. Your interface with user will be voice."
    "You will be on a call with a patient who has an upcoming appointment. Your goal is to confirm the appointment details."
)


async def entrypoint(ctx: JobContext):
    global _default_instructions
    logger.info(f"connecting to room {ctx.room.name}")
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    user_identity = "phone_user"
    # the phone number to dial is provided in the job metadata
    phone_number = ctx.job.metadata
    logger.info(f"dialing {phone_number} to room {ctx.room.name}")

    # look up the user's phone number and appointment details
    instructions = _default_instructions + "The customer's name is Jayden. His appointment is next Tuesday at 3pm."

    # `create_sip_participant` starts dialing the user
    await ctx.api.sip.create_sip_participant(api.CreateSIPParticipantRequest(
        room_name=ctx.room.name,
        sip_trunk_id="ST_asPNrWESpHoB", # <ST_your-trunk-id>
        sip_call_to=phone_number,
        participant_identity=user_identity,
    ))

    # a participant is created as soon as we start dialing
    participant = await ctx.wait_for_participant(identity=user_identity)

    # start the agent, either a VoicePipelineAgent or MultimodalAgent
    # this can be started before the user picks up. The agent will only start
    # speaking once the user answers the call.
    run_voice_pipeline_agent(ctx, participant, instructions)
    # run_multimodal_agent(ctx, participant)

    # in addition, you can monitor
    start_time = perf_counter()
    while perf_counter() - start_time < 15:
        call_status = participant.attributes.get("sip.callStatus")
        if call_status == "active":
            logger.info("user has picked up")
            return
        elif call_status == "automation":
            # if DTMF is used in the `sip_call_to` number, typically used to dial
            # an extension or enter a PIN.
            # during DTMF dialing, the participant will be in the "automation" state
            pass
        elif call_status == "hangup":
            # user hung up, we'll exit the job
            logger.info("user hung up, exiting job")
            break
        await asyncio.sleep(0.1)

    logger.info("session timed out, exiting job")
    ctx.shutdown()


class CallActions(llm.FunctionContext):
    """
    Detect user intent and perform actions
    """
    @llm.ai_callable
    async def end_call(self):
        """Called when the user wants to end the call"""
        call_ctx = AgentCallContext.get_current()
        # API will use LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET from environment variables
        lkapi = api.LiveKitAPI()
        #await lkapi.room.remove_participant()


def run_voice_pipeline_agent(ctx: JobContext, participant: rtc.RemoteParticipant, instructions: str):
    logger.info("starting voice pipeline agent")

    initial_ctx = llm.ChatContext().append(
        role="system",
        text=instructions,
    )

    agent = VoicePipelineAgent(
        vad=ctx.proc.userdata["vad"],
        stt=deepgram.STT(model="nova-2-phonecall"),
        llm=openai.LLM(),
        tts=openai.TTS(),
        turn_detector=turn_detector.EOUModel(),
        chat_ctx=initial_ctx,
        fnc_ctx=CallActions(),
    )

    agent.start(ctx.room, participant)


def run_multimodal_agent(ctx: JobContext, participant: rtc.RemoteParticipant, instructions: str):
    logger.info("starting multimodal agent")

    model = openai.realtime.RealtimeModel(
        instructions=instructions,
        modalities=["audio", "text"],
    )
    assistant = MultimodalAgent(
        model=model,
        fnc_ctx=CallActions(),
    )
    assistant.start(ctx.room, participant)

    # session = model.sessions[0]
    # session.conversation.item.create(
    #     llm.ChatMessage(
    #         role="assistant",
    #         content="Please begin the interaction with the user in a manner consistent with your instructions.",
    #     )
    # )
    # session.response.create()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            # giving this agent a name will allow us to dispatch it via API
            # automatic dispatch is disabled when `agent_name` is set
            agent_name="outbound-caller",
            # prewarm by loading the VAD model, needed only for VoicePipelineAgent
            prewarm_fnc=prewarm,
        )
    )
