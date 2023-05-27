import wordninja

def Normalize(text):
    # Remove spaces
    text = text.replace(" ", "")

    # Split the sentences based on commas and periods
    sentences = []
    sentence = ""
    temp = ""
    for i in range(len(text)):
        temp += text[i]
        if text[i] == ",":
            result = wordninja.split(temp)
            sentence += " ".join(result)
            sentence += ", "
            temp = ""
        if text[i] == ".":
            result = wordninja.split(temp)
            sentence += " ".join(result)
            sentence += "."
            sentences.append(sentence)
            sentence = ""
            temp = ""
    # If there is no punctuation
    if sentences == []:
        result = wordninja.split(temp)
        sentence += " ".join(result)
        sentences.append(sentence)

    return sentences

