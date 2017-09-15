import sys
def main(argv):
    file_input = open(argv[0],'r')
    text_input = file_input.read()
    file_input.close()
    words_list_all = text_input.split( )
    words_list = []
    words_count = []
    text_output = ""

    for word in words_list_all:
        if word not in words_list:
            words_list.append(word)
            words_count.append(0)

    for i in range(len(words_list)):
        for word in words_list_all:
            if word == words_list[i]:
                words_count[i] += 1
    
    file_output = open('Q1.txt','w')
    for i in range(len(words_list)):
        if i == len(words_list)-1: # the last one
            file_output.write("{0} {1} {2}".format(words_list[i], i, words_count[i]))
        else:
            file_output.write("{0} {1} {2}\n".format(words_list[i], i, words_count[i]))
   
    file_output.close()

if __name__ == '__main__':
    main(sys.argv[1:])
