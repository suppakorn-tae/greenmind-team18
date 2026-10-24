# E031 — thinking budget 3072 -> 2048
Parent E030. One variable. Why: 3072 beat 5120 with no accuracy loss; 2048 cuts up to 11% of generated tokens (chat 37/90 and reason 9/45 final requests exceed 2048 completion tokens). Risk: reason/chat accuracy. Gate: A=1 and reason acc >= 0.93 on final. Start 2026-10-24T19:26:33+07:00.
