"""Text-to-Speech (TTS) engine for subvocal feedback.

Supports native macOS 'say' / 'afplay' utilities, OpenAI Audio API, and pyttsx3.
"""

import os
import sys

import json
import subprocess
import urllib.request
import urllib.error
from typing import Optional

from subvocal.tts.schema import TTSConfig
import logging

logger = logging.getLogger(__name__)


class TTSEngine:
    """Multi-backend Text-to-Speech generator and player."""

    def __init__(self, config: Optional[TTSConfig] = None):
        self.config = config or TTSConfig()
        
        # Ensure output directory exists
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Detect platform and credentials
        self.is_mac = sys.platform == "darwin"
        self.has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        
    def speak(self, text: str, filename: Optional[str] = None) -> str:
        """Synthesize and speak the text, saving the audio to a file.
        
        Args:
            text: The plain text statement to speak.
            filename: Target file name (optional). If omitted, a file name
                      is auto-generated based on text hash.
                      
        Returns:
            The absolute path of the generated audio file.
        """
        if not text:
            return ""
            
        clean_text = text.strip()
        logger.info("Synthesizing: %r", clean_text)
        
        # Generate safe file name
        if not filename:
            safe_text = "".join(c for c in clean_text[:20] if c.isalnum() or c in (" ", "_")).strip()
            safe_text = safe_text.replace(" ", "_").lower() or "tts_output"
            filename = f"{safe_text}.{self.config.audio_format}"
            
        output_path = os.path.join(self.config.output_dir, filename)
        abs_output_path = os.path.abspath(output_path)
        
        # 1. Try OpenAI TTS first if key is available
        if self.has_openai:
            success = self._synthesize_openai(clean_text, abs_output_path)
            if success:
                self._play_audio_file(abs_output_path)
                return abs_output_path
                
        # 2. Try macOS Native say/afplay utility if on Mac
        if self.is_mac:
            success = self._synthesize_macos(clean_text, abs_output_path)
            if success:
                self._play_audio_file(abs_output_path)
                return abs_output_path
                
        # 3. Try pyttsx3 offline library
        success = self._synthesize_pyttsx3(clean_text, abs_output_path)
        if success:
            return abs_output_path
            
        # 4. Fallback to console warning if no TTS works
        logger.warning("Playing offline fallback audio simulation: (Beep) %r", clean_text)
        return ""

    def _synthesize_openai(self, text: str, dest_path: str) -> bool:
        """Call OpenAI Audio synthesis API via urllib."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return False
            
        logger.info("Attempting OpenAI Cloud TTS...")
        url = "https://api.openai.com/v1/audio/speech"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Mapping configurations
        voice = self.config.voice if self.config.voice in ("alloy", "echo", "fable", "onyx", "nova", "shimmer") else "alloy"
        
        data = json.dumps({
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "speed": self.config.speed
        }).encode("utf-8")
        
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as response:
                with open(dest_path, "wb") as f:
                    f.write(response.read())
            logger.info("OpenAI TTS synthesis success: %s", dest_path)
            return True
        except urllib.error.URLError as e:
            logger.error("OpenAI API call failed: %s", e)
        except Exception as e:
            logger.error("OpenAI TTS general failure: %s", e)
            
        return False

    def _synthesize_macos(self, text: str, dest_path: str) -> bool:
        """Synthesize audio using macOS native `say` terminal command."""
        logger.info("Attempting macOS native 'say' utility...")
        try:
            # macOS native say defaults to AIFF. If wav is requested, we can use say's wave export
            # syntax: say -o output.wav --data-format=LEI16@22050
            # Let's adjust command arguments based on format config
            rate = str(int(175 * self.config.speed)) # Default speaking rate is 175 wpm
            
            cmd = ["say", "-r", rate]
            
            # Select specific voice if configured and not matching OpenAI voice list
            if self.config.voice not in ("alloy", "echo", "fable", "onyx", "nova", "shimmer", ""):
                cmd.extend(["-v", self.config.voice])
                
            cmd.extend(["-o", dest_path])
            
            # For macOS say, appending format is required for correct wav output
            if self.config.audio_format == "wav":
                cmd.extend(["--data-format=LEI16@22050"])
            elif self.config.audio_format == "mp3":
                # macOS 'say' doesn't support mp3 direct output. It will save as AAC or AIFF.
                # Let's override output format extension to wav/aiff for local say command
                dest_path_corrected = dest_path.rsplit(".", 1)[0] + ".wav"
                cmd[-1] = dest_path_corrected
                cmd.extend(["--data-format=LEI16@22050"])
                dest_path = dest_path_corrected
                
            cmd.append(text)
            
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info("macOS native TTS synthesis success: %s", dest_path)
            return True
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.error("macOS native 'say' utility failed: %s", e)
        return False

    def _synthesize_pyttsx3(self, text: str, dest_path: str) -> bool:
        """Fallback to offline pyttsx3 library."""
        try:
            import pyttsx3
            logger.info("Attempting pyttsx3 offline TTS library...")
            engine = pyttsx3.init()
            
            # Configure speed
            rate = engine.getProperty("rate")
            engine.setProperty("rate", int(rate * self.config.speed))
            
            # Save to file
            engine.save_to_file(text, dest_path)
            engine.runAndWait()
            
            # Speak aloud
            engine.say(text)
            engine.runAndWait()
            
            logger.info("pyttsx3 synthesis success: %s", dest_path)
            return True
        except ImportError:
            # pyttsx3 package is not installed
            pass
        except Exception as e:
            logger.error("pyttsx3 offline TTS failed: %s", e)
        return False

    def _play_audio_file(self, file_path: str):
        """Play generated audio file locally using native system commands."""
        if not os.path.exists(file_path):
            return
            
        if self.is_mac:
            try:
                # Play using macOS native afplay tool
                subprocess.Popen(["afplay", file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                logger.error("Local audio playback failed: %s", e)
        else:
            # Non-mac offline playback could be implemented with play/aplay if needed
            pass


if __name__ == "__main__":
    # Test script execution
    print("Testing Subvocal Text-to-Speech Engine...")
    engine = TTSEngine(TTSConfig(voice="Samantha", audio_format="wav"))
    file_path = engine.speak("System initialized. Subvocal connection established.")
    print(f"Generated file location: {file_path}")
