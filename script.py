import random

random.seed(2)

data = []
def divide_list(lst):
    # Calculate the index corresponding to 70% of the list length
    index = int(len(lst) * 0.7)

    # Split the list into two parts
    first_part = lst[:index]
    second_part = lst[index:]

    return first_part, second_part

def tek_cif(liste):
    first_part = []
    second_part = []
    print(liste)
    for i in range(len(liste)):
        if  liste[i][0]*10 % 2 == 0 and liste[i][1]*10 % 2 == 0:
            second_part.append(liste[i])
        elif liste[i][0]*10 % 2 != 0 and liste[i][1]*10 % 2 != 0 :
            first_part.append(liste[i])
            print(liste[i])
    return first_part, second_part
        


def write_list_to_file(lst, filename1, filename2):
    # Open the file in write mode
    with open(filename1, 'w') as f:
        with open(filename2, 'w') as g:
            # Loop through the list
            for item in lst:
                # Write each item to the file, followed by a newline
                f.write( "["+str(item[0]) + "," + str(item[1]) + '],')
                g.write( "["+ str(item[2]/100) + '],')

def write_list_to_file2(lst, filename1, filename2):
    # Open the file in write mode
    with open(filename1, 'w') as f:
        with open(filename2, 'w') as g:
            # Loop through the list
            for item in lst:
                # Write each item to the file, followed by a newline
                f.write( "["+str(item[0]) + "," + str(item[1]) + '],')
                

for x in range(1,11):
    for y in range(1,11):
        data.append([x/10,y/10,x*y])

shuffled_data = sorted(data, key=lambda x: random.random())

training, testing = tek_cif(shuffled_data)



write_list_to_file(training, "training_input.txt","training_output.txt")
write_list_to_file2(testing, "testing_input.txt","testing_output.txt")


with open("check.txt", 'w') as c:
    # Loop through the list
    for item in testing:
        # Write each item to the file, followed by a newline
        c.write( str(int(item[2])) + ',')


