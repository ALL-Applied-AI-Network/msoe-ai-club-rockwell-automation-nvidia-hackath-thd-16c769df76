# Carryover Monitor video shot list

The nine narration clips total approximately **5:06**. Times below use the clips in numeric order with no added gaps. Allow a few frames of room at each edit.

## Timeline

| Video time | Audio | Required visuals |
|---|---|---|
| 0:00–0:25 | 1. Introduction | 3–4 second title card; wide CSI fluid-line image or footage; close view of tanks/shared dispensing path; vial/nozzle close-up. Add a simple label: `Problem: first-vial carryover after recipe changes`. |
| 0:25–0:57 | 2. Our Solution | Clean flow diagram: `Recipe queue + vial camera → DGX Spark model → operator dashboard`; brief dashboard overview; highlight future-risk warning and visual confirmation as separate stages. |
| 0:57–1:41 | 3. Testing Data | 30 fps synthetic/digital-twin vial clip; montage of varied liquid colors and defect severities; recipe CSV with tank volumes and expected color/hue; real Dataset 5 clip; simple before/after timing-alignment graphic. |
| 1:41–2:10 | 4. Physical-Line Testing Limitation | CSI line photo; two short cards: `Visit 1 — PLC machine unavailable` and `Visit 2 — authorized line-Spark access unavailable`; close on terminal/Spark photo only after cropping or blurring usernames, IP addresses, and other access details. End with `No access controls bypassed`. |
| 2:10–2:45 | 5. Recipe-Queue Limitation | Manually prepared recipe queue CSV; dashboard queue populated from it; comparison graphic: `Hackathon: queue prepared ahead of time` versus `Production: PLC → approved read-only interface → monitor`. |
| 2:45–3:44 | 6. Dashboard Demonstration | Record the port-20401 Saved Run Demo at readable zoom. Capture Start Demo, current vial/queue movement, live vial frame, and examples of orange, purple, blue, and red events. Use multiple real-time excerpts if events are separated by minutes; label the sequence `Real-time replay — edited excerpts`. Do not imply the edited minute is one continuous run. |
| 3:44–4:15 | 7. Measured Results | Results slide: `23 vials / 7 expected carryover cases`; at balanced threshold 0.0265: `6/7 detected, 2 false alarms, ~87% accuracy`; `AUC 0.839`; `4.1 ms mean inference`. Separate callout: `Demo threshold 0.0071 — prioritizes recall; more false alarms expected`. |
| 4:15–4:37 | 8. Manufacturing Value | Four simple icons or cards: less scrap, less rework, less manual inspection, targeted flushing/reduced downtime. Follow with a short industry list: paint, chemical, pharmaceutical, food and beverage, personal care. |
| 4:37–5:06 | 9. Limitations and Conclusion | Split view of recorded CSI footage and dashboard; concise limitation cards: small dataset, no untouched live trial, prerecorded camera data, manually entered queue. Then future-work flow: live camera + read-only PLC integration + held-out validation. Finish with project name, team name, and GitHub link/QR code. |

## Footage to collect

1. **Dashboard capture — essential**
   - Record one complete Dataset 5 demonstration so all event types are available for editing.
   - Capture at 1080p or higher with browser zoom large enough to read the queue and event labels.
   - Do not show the dashboard token, SSH windows, passwords, or private keys.
   - Keep the screen recording at normal speed. Edit together labeled excerpts for the final one-minute dashboard section.

2. **CSI line visuals — essential**
   - One wide shot or photograph of the fluid-mixing line.
   - One closer view of the filling/nozzle area if available.
   - Existing CSI terminal photographs can support the access-limitation section, but crop or blur usernames, hostnames, IP addresses, and terminal commands.

3. **Data visuals — essential**
   - Two or three synthetic vial-fill clips showing different colors or carryover severity.
   - A short real Dataset 5 vial-fill excerpt.
   - A clean screenshot of the generated recipe CSV.
   - A small alignment graphic showing original timing drift and corrected fill windows.

4. **Graphics to create**
   - System flow diagram.
   - Manual-versus-production recipe-queue diagram.
   - Results slide with the threshold distinction.
   - Manufacturing-value slide.
   - Limitations/future-work slide.
   - Opening title and closing GitHub/credits card.

## Editing notes

- Keep visual cuts aligned with narration topics rather than changing shots every sentence.
- Use short text overlays; the narration carries the explanation.
- Distinguish synthetic footage, recorded real CSI footage, and real-time replay with visible labels.
- Keep the `0.0265` evaluation metrics separate from the live dashboard's `0.0071` recall-prioritized operating threshold.
- Leave roughly 0.25 seconds between audio clips if the transitions sound abrupt. This may bring the finished video to approximately 5:08 without exceeding the requested approximate length materially.

## First-cut implementation

The first complete draft is rendered at **1920x1080, 30 fps**, with a duration of **5:09.8**.

- Introduction: title card followed by CSI line and filling-station footage.
- Solution: architecture diagram.
- Testing data: testing-data slide and a 30 fps digital-twin vial fill.
- Physical-line limitation: limitation slide, CSI HMI footage, and tank photograph.
- Recipe queue: manual-versus-read-only PLC diagram.
- Dashboard: edited real-time excerpts from the Recipe 6-to-7 transition, including the future-risk state and confirmed-carryover popup.
- Results: balanced-threshold metrics, with the 0.0071 demo threshold shown separately.
- Manufacturing value: value graphic and a wider CSI robot/line view.
- Conclusion: limitations/next-steps graphic and recorded CSI vial-camera footage.

Narration was normalized to approximately **-16.6 LUFS**. The unmodified source files and intermediate section renders remain preserved for revisions.
