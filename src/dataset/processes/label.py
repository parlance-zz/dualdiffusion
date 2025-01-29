# MIT License
#
# Copyright (c) 2023 Christopher Friesen
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

#todo: label process tbd, old code below as reference

"""
    if tags is not None: 
        tag_labels = {tag: [tag] for tag in tags}
        if labels is not None:
            labels.update(tag_labels)
        else:
            labels = tag_labels

    if labels is not None:

        print(f"Calculating text embeddings for {len(labels)} labels...")
        label_embeddings = []
        for label, tag_list in labels.items():
            if isinstance(tag_list, str):
                tag_list = [tag_list]
            elif not isinstance(tag_list, list):
                raise ValueError(f"Invalid tag list for label '{label}': {tag_list}")
            
            tag_embeddings = clap_model.get_text_embedding(tag_list, use_tensor=True)
            label_embeddings.append(tag_embeddings.mean(dim=0))
        label_embeddings = torch.stack(label_embeddings, dim=0).to(device=device, dtype=torch.float32)
        label_embeddings = normalize(label_embeddings).float()
    
        # get text embeddings if a prompt is available and they are not yet encoded
        if sample_prompt is not None and text_embeddings is None:
            save_latents = True
            text_embeddings = normalize(clap_model.get_text_embedding([sample_prompt], use_tensor=True)).float()

        if labels is not None:
            # gets similarity for each label and chunk individually
            #cos_similarity = torch.mm(label_embeddings / label_embeddings.shape[1]**0.5,
            #                          audio_embeddings.T / audio_embeddings.shape[1]**0.5).clip(-1, 1)

            # update audio file metadata with label similarity scores
            label_scores = (torch.einsum("ld,d->l", label_embeddings, audio_embeddings.mean(dim=0)) / label_embeddings.shape[1])
            label_scores = ((label_scores + 1) / 2).clip(0, 1).tolist() # keep score positive to preserve textual sorting order
            labels_metadata = {f"clap_{label}": f"{score:01.4f}" for label, score in zip(labels.keys(), label_scores)}
            labels_metadata["clap_all_labels"] = f"{sum(label_scores) / len(label_scores):01.4f}"
"""