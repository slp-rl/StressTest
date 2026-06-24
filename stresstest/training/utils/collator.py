import librosa
import torch


class StressDataCollator:
    def __init__(self, processor, max_length: int = None):
        self.processor = processor
        self.max_length = max_length or processor.feature_extractor.n_samples
        self.sampling_rate = processor.feature_extractor.sampling_rate

    def resample_audio(self, audio, sr): 
        return librosa.resample(y=audio, target_sr=self.sampling_rate, orig_sr=sr)
    
    def prepare_audio(self, audio):
        return self.resample_audio(audio=audio['array'], sr=audio['sampling_rate'])
    
    def get_conversation(self, question_text, audio_url, answer_text):
        return [
                {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_url},
                {"type": "text", "text": question_text},
            ]},
            {"role": "assistant", "content": answer_text}
        ]
    
    def mask_labels_tokens(self, tokens, pattern):
        # create a mask initialized as False
        mask = torch.zeros_like(tokens, dtype=torch.bool)

        for i, sample in enumerate(tokens):
            # Find matches for the pattern
            for j in range(len(sample) - len(pattern) + 1):
                if torch.equal(sample[j:j+len(pattern)], pattern):
                    # Mark everything before the end of the pattern
                    mask[i, :j+len(pattern)] = True
                    break  # Stop after finding the first occurrence

        # apply the mask: tokens before and including the pattern become -100
        tokens_masked = tokens.masked_fill(mask, -100)

        return tokens_masked

    def __call__(self, examples):
        audios = []
        qa_texts = [] # question and answer processed texts
        
        for example in examples:
            try:
                audio = self.prepare_audio(example['audio'])
                audios.append(audio)
                question = example['question']
                answer = example['answer']
                audio_id = example['audio_id']
                conversation = self.get_conversation(question, audio_id, answer)
                qa_text = self.processor.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
                qa_texts.append(qa_text)
            except Exception as e:
                print(f"Error processing example: {e}")
                raise e

        try:
            inputs = self.processor(text=qa_texts, audios=audios, return_tensors="pt", sampling_rate=self.processor.feature_extractor.sampling_rate, padding=True)
        except Exception as e:
            print(f"Error processing inputs: {e}")
            raise e
        
        # prepare the labels
        labels = inputs['input_ids'].clone()
        # Masking labels individually per example with the assistant token, replacing token up to it with -100
        assistant_token_ids = self.processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)
        masked_labels = self.mask_labels_tokens(labels, torch.tensor(assistant_token_ids))

        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "input_features": inputs.input_features,
            "feature_attention_mask": inputs.feature_attention_mask,
            "labels": masked_labels,
        }