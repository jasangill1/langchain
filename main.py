from flask import Flask, request, jsonify
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import openai
import os

app = Flask(__name__)

openai.api_key = os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0.5)
prompt = PromptTemplate(
    input_variables=["message"],
    template="{message}",
)
chain = LLMChain(llm=llm, prompt=prompt)


@app.route('/generate', methods=['POST'])
def generate():
    message = request.json['message']

    try:
        response = chain.run(message)
        return jsonify({'completion': response})

    except Exception as error:
        print(f"Error generating completion: {error}")
        return jsonify({'error': 'Error generating completion'}), 500


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port, debug=True)
