use espeak_ng_sys as espeak_ng;
use std::ffi::{CStr, CString};
use std::ptr;

pub struct Espeak {
    _private: (),
}

impl Espeak {
    pub fn new(data_path: Option<&str>) -> anyhow::Result<Self> {
        let c_path = data_path.map(|s| CString::new(s).unwrap());

        let path_ptr = c_path.as_ref().map(|s| s.as_ptr()).unwrap_or(ptr::null());

        // output = Retrieval, buflength = 0, options = 0
        let sample_rate = unsafe {
            espeak_ng::espeak_Initialize(
                espeak_ng::espeak_AUDIO_OUTPUT_AUDIO_OUTPUT_RETRIEVAL,
                0,
                path_ptr,
                0,
            )
        };

        if sample_rate <= 0 {
            anyhow::bail!("Failed to initialize eSpeak-NG (returned {})", sample_rate);
        }

        Ok(Espeak { _private: () })
    }

    pub fn set_voice(&self, name: &str) -> anyhow::Result<()> {
        let c_name = CString::new(name)?;
        let result = unsafe { espeak_ng::espeak_SetVoiceByName(c_name.as_ptr()) };
        if result == espeak_ng::espeak_ERROR_EE_OK {
            Ok(())
        } else {
            anyhow::bail!("Failed to set eSpeak voice to {}", name)
        }
    }

    pub fn text_to_phonemes(&self, text: &str) -> anyhow::Result<String> {
        let c_text = CString::new(text)?;
        let mut text_ptr = c_text.as_ptr() as *const std::os::raw::c_char;
        let mut all_phonemes = String::new();

        while !text_ptr.is_null() && unsafe { *text_ptr != 0 } {
            let initial_ptr = text_ptr;
            let mut current_ptr = text_ptr as *const std::os::raw::c_void;

            // textmode 1 = UTF8, phonememode 66 = IPA + Punctuation
            let result_ptr = unsafe { espeak_ng::espeak_TextToPhonemes(&mut current_ptr, 1, 66) };

            if !result_ptr.is_null() {
                let c_str = unsafe { CStr::from_ptr(result_ptr) };
                let phonemes = c_str.to_string_lossy();
                if !all_phonemes.is_empty()
                    && !all_phonemes.ends_with(' ')
                    && !phonemes.starts_with(' ')
                {
                    all_phonemes.push(' ');
                }
                all_phonemes.push_str(&phonemes);
            }

            text_ptr = current_ptr as *const std::os::raw::c_char;

            if text_ptr.is_null() || text_ptr == initial_ptr {
                break;
            }
        }

        if all_phonemes.is_empty() {
            anyhow::bail!("eSpeak failed to generate phonemes");
        }

        Ok(all_phonemes)
    }
}

impl Drop for Espeak {
    fn drop(&mut self) {
        unsafe {
            espeak_ng::espeak_Terminate();
        }
    }
}
