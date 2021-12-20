
def isFrameSecure(model_list, labels, bounding_boxes, class_numbers, results_list):
    ''' function responsible for for checking security integrety in each frame.
    Will return True if the frame is secure, and false if not

    More information:

    models:
    0 - People model >> [ 0 - Person ]
    1 - hat model  >> [ 0 - without_hat, 1 - with_hat, 2 - Person ]
    2 - mask model >> [ 0 - without_mask, 1 - with_mask, 2 - mask_weared_incorrect ]
    
    model_list - list that contains info about what model we are currently looking at.
    
    labels - list of lists. Each list is the labels of the corresponding class.
    example >> labels of model 1 - labels[1] = [ without_hat, with_hat, Person ]   
    
    bounding_boxes - list that contain the location of the object detected [ x, y, width, height ].
    
    class_numbers - list of the class in the corresponding model.
    example >>
    class_numbers = [ 0 ]
    model_list    = [ 1 ]

    in model 1 the class 0 is = without_hat >> labels[1][0] = "without_hat"

    results_list = list containing the index of the objects that have passed the Non-maximum suppression.
    Only the objects in this list will count for our algorithm the end.
    
    linked lists - model_list, bounding_boxes, class_numbers
    
    '''

    # -----------------------------------------------------------------
    #         First we will separate objects into 
    #         each category that interests us
    # -----------------------------------------------------------------

    #                                   [   0   ,   1    ,  2   ]
    # in this case we have >> classes = [ person, hardhat, mask ]

    people = []
    no_hats = []
    no_masks = []

    # i is the object identity inside the lists
    for i in range(len(model_list)):

        # Checking if object is in the final result, after the KNN algorithm
        if i in results_list:

            # check for people
            if model_list[i] == 0 :
                people.append(i)

            # check for hats
            elif model_list[i] == 1 and class_numbers[i] == 0:
                no_hats.append(i)

            # check for mask
            elif model_list[i] == 2 and ( class_numbers[i] == 0 or class_numbers[i] == 2):
                no_masks.append(i)

    # We should now have 3 lists filled
    
    # print(people)
    # print(no_hats)
    # print(no_masks)

    # -----------------------------------------------------------------
    #         Now we have to verify interception on a 2d plane to
    #         identify if each person is using the equipment
    # -----------------------------------------------------------------

    ok_list =[]
    no_list =[]

    # looping through all people to check security
    for person in people:
        # person = index of the linked list

        is_hat_ok = True
        is_mask_ok = True

        for hat in no_hats:
            intersect = overlap(bounding_boxes[hat], bounding_boxes[person])
            if intersect:
                is_hat_ok = False
                break
        for mask in no_masks:
            intersect = overlap(bounding_boxes[mask], bounding_boxes[person])
            if intersect:
                is_mask_ok = False
                break

        if is_hat_ok and is_mask_ok:
            ok_list.append(person)
        else:
            no_list.append(person)
            # print(f'Person {person}:')
            # if not is_hat_ok:
            #     print(' - not using their Hat')
            # if not is_mask_ok:
            #     print(' - not using their mask')


    # -----------------------------------------------------------------
    #     Now we have a list of people that are ok and that are not
    #     so we log the results and return True or False
    # -----------------------------------------------------------------

    print('-' * 30)
    print(f'people that are ok: {ok_list}')
    print(f'people that are not ok: {no_list}')
    print('-' * 30)

    if len(no_list) != 0:
        return False
    else:
        return True

def overlap(rectangle1, rectangle2):
    rect1 = [rectangle1[0], rectangle1[1], rectangle1[0] + rectangle1[2], rectangle1[1] + rectangle1[3]]
    rect2 = [rectangle2[0], rectangle2[1], rectangle2[0] + rectangle2[2], rectangle2[1] + rectangle2[3]]

    #        [0 , 1, 2, 3]
    # rect = [x1,y1,x2,y2]

    if (rect2[2] > rect1[0] and rect2[2] < rect1[2]) or (rect2[0] > rect1[0] and rect2[0] < rect1[2]) or \
            (rect2[0] < rect1[0] and rect2[2] > rect1[2]):
        x_match = True
    else:
        x_match = False
    if (rect2[3] > rect1[1] and rect2[3] < rect1[3]) or (rect2[1] > rect1[1] and rect2[1] < rect1[3]) or (
            rect2[1] < rect1[1] and rect2[3] > rect1[3]):
        y_match = True
    else:
        y_match = False

    if x_match and y_match:
        return True
    else:
        return False


# test objects

# person0 = [1,1,1,2]
# person1 = [3,1,1,2]
# person2 = [5,1,1,2]
#
# hat3 = [1.1,0.9,0.5,0.5]
# hat4 = [5.1,0.1,0.5,0.5]
#
# mas5 = [1.1,1.1,0.2,0.2]
# mas6 = [3.1,1.1,0.2,0.2]
#
# classes = [ 'person', 'hardhat', 'mask']
# bounding_boxes = [person0,person1,person2,hat3,hat4,mas5,mas6]
# confidences = [1,1,1,1,1,1,1]
# class_numbers = [0,0,0,1,1,2,2]
# results = [0,1,2,3,4,5,6]
#
# print('Is the frame Secure?         '+str(isFrameSecure(classes,bounding_boxes,confidences,class_numbers,results)))

# print(overlap(person0,hat3))
# print(overlap(bounding_boxes[3],bounding_boxes[1]))

# isFrameSecure(model_list, labels, bounding_boxes, class_numbers, results_list)

model_list_example = [0, 0, 1, 1, 1, 1, 2, 2, 2]
labels_example = [['person'], ['without_hat', 'with_hat', 'Person'], ['without_mask', 'with_mask', 'mask_weared_incorrect']]
bounding_boxes = [[362, 201, 114, 621], [361, 218, 114, 627], [451, 213, 29, 139], [451, 214, 29, 137],
                  [453, 209, 27, 142], [452, 208, 28, 142], [468, 259, 12, 89], [468, 257, 12, 91], [466, 259, 13, 88]]
class_numbers = [0, 0, 1, 1, 1, 1, 0, 0, 0]
results_list = [1,7,4]

print()

print(isFrameSecure(model_list_example,labels_example,bounding_boxes,class_numbers,results_list))




