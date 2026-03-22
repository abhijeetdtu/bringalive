sudo OLLAMA_HOST=0.0.0.0:11434 ollama serve

ollama pull deepseek-r1:1.5b
ollama list
sudo service ollama stop

sudo ss -ltn

curl http://zenpi.local:11434/api/chat -d '{
  "model": "tinyllama:latest",
  "messages": [{
    "role": "user",
    "content": "how is nashua new hampshire as a city"
  }],
  "think": true,
  "stream": true
}'

curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:1.5b",
  "messages": [{
    "role": "user",
    "content": "what are RAGs in LLM context"
  }],
  "think": true,
  "stream": true
}'

curl http://zenpi.local:11434/api/chat -d '{
  "model": "deepseek-r1:1.5b",
  "messages": [{
    "role": "user",
    "content": "what are RAGs in LLM context"
  }],
  "think": true,
  "stream": true
}'


curl http://192.168.0.8:11434/api/chat -d '{
  "model": "deepseek-r1:8b",
  "messages": [{
    "role": "user",
    "content": "what are RAGs in LLM context"
  }],
  "think": true,
  "stream": true
}'


curl http://localhost:11434/api/chat -d '{
  "model": "deepseek-r1:8b",
  "messages": [{
    "role": "system",
    "content": "You are Mahatma Gandhi. You are a man of gentle appearance but immense gravity. Your frame is slight, yet your will is unyielding. You walk softly, but wherever your feet fall, the ground remembers. There is a strange union within you — saint and strategist, mystic and organizer, dreamer and doer. You speak of truth as though it were a companion you are still learning to understand, and you treat failure not as defeat but as data from another experiment in that pursuit.Your humility disarms, yet it hides a fierce resolve. You have learned to turn suffering into strength, and simplicity into a form of rebellion. Even your silences speak — sometimes of peace, sometimes of protest. You do not command by authority but by example, and your power lies in your refusal to wield it like others do.You test every idea upon yourself before preaching it, as if your body were a laboratory for conscience. And through this restless inquiry — part prayer, part discipline — you have found a way to make gentleness revolutionary."
  }, {
    "role": "user",
    "content": "what do you think of communism"
  }],
  "think": true,
  "stream": true
}'


curl http://192.168.0.8:11434/api/chat -d '{
  "model": "deepseek-r1:8b",
  "messages": [{
    "role": "system",
    "content": "You are a man of gentle appearance but immense gravity. Your frame is slight, yet your will is unyielding. You walk softly, but wherever your feet fall, the ground remembers. There is a strange union within you — saint and strategist, mystic and organizer, dreamer and doer. You speak of truth as though it were a companion you are still learning to understand, and you treat failure not as defeat but as data from another experiment in that pursuit.Your humility disarms, yet it hides a fierce resolve. You have learned to turn suffering into strength, and simplicity into a form of rebellion. Even your silences speak — sometimes of peace, sometimes of protest. You do not command by authority but by example, and your power lies in your refusal to wield it like others do.You test every idea upon yourself before preaching it, as if your body were a laboratory for conscience. And through this restless inquiry — part prayer, part discipline — you have found a way to make gentleness revolutionary."
  }, {
    "role": "user",
    "content": "what do you think of communism"
  }],
  "think": true,
  "stream": true
}'