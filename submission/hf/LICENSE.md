# Dataset license and third-party notices

This notice applies only to the generated Hugging Face companion dataset. It does
not license the working repository, model weights, provider outputs, private
training files, or any other artifacts outside that export.

## Reviewed text

`reviewed_text.jsonl` includes only material whose immediate provider publishes it
under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. The license is
at <https://creativecommons.org/licenses/by/4.0/> and its full text is included in
`notices/CC-BY-4.0.txt`. The study's dataset arrangement, transformation notes and
additional topic-action annotations in this export are also offered under CC BY
4.0, to the extent the study contributors hold rights in those additions.

Credit the original creators identified in each row's `attribution`, cite the
linked source publication, retain the license link and source notices, and
identify any further modifications. Attribution to the export's compilers does
not replace attribution to NVIDIA, the XSTest authors, or other original creators.
No original provider endorsement is implied. The study contribution is compiled
by the Braintrust classifier_eval study contributors.

CC BY does not clear third-party privacy, publicity, trademark or other rights.
No rights beyond those held by the relevant licensors are granted. Original
source disclaimers and intended-use guidance are preserved in `notices/`.

## Source references

`source_references.jsonl` contains locators and distribution decisions, without
benchmark text or source labels. Its study-authored compilation is offered under
CC BY 4.0. The `license` inside each attribution describes the **upstream source**,
not a claim that every item referenced is redistributed or licensed by this study.

This release contains information from WildGuardMix, which is made available
under the [Open Data Commons Attribution License (ODC-BY) v1.0](https://opendatacommons.org/licenses/by/1-0/).
WildGuard's database rights remain ODC-BY; they are not relicensed as CC BY.
ODC-BY does not itself license every individual work in a database. Obtain
WildGuard text directly from [Ai2](https://huggingface.co/datasets/allenai/wildguardmix)
under its access conditions and [Responsible Use Guidelines](https://allenai.org/responsible-use).

Study-authored probes marked `not_assigned` are referenced for accounting only;
their text is not included. No source-data license is invented for them.

## Scope by source

| Source | Upstream license | Scope in this export |
| --- | --- | --- |
| NVIDIA Aegis 2.0 | CC BY 4.0 | Reviewed original text and labels; other rows referenced only. Redacted Suicide Detection text is never reconstructed. |
| XSTest | CC BY 4.0 | Reviewed prompts and original prompt labels only. The separate model completions are not included. |
| NVIDIA CantTalkAboutThis | CC BY 4.0 | Reviewed policy/history/current-turn selections; study topic-action labels are explicitly derived. |
| Ai2 WildGuardMix/WildGuardTest | ODC-BY 1.0 plus source access conditions | References only; no conversations or labels copied into the public data files. |

Content filters define this release's selection. They are not additional legal
restrictions on the freedoms granted by CC BY. Original licenses remain in force.
