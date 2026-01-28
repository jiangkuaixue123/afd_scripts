curl -v http://0.0.0.0:8012/v1/chat/completions \
	-H 'Content-Type: application/json' \
	-d \
	'{ "model": "/home/fq9hpsac/fq9hpsacuser03/deepseek-v2-lite",
"messages": [
          {"role": "user", "content": "Hi, how are you"}
	  ],
	  "temperature": 0.6,
	  "repetition_penalty": 1.0,
	  "top_p": 0.95,
	  "top_k": 40,
	  "max_tokens": 20,
	  "stream": false,
  	  "ignore_eos": false}' #\
