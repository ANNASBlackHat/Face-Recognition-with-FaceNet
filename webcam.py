import cv2
import face_detection_cv2 as face_cv2
import face_detection_tf as face_tf
import sys, os, getopt
import time


def read_args(argv):
    try:
        opts, args = getopt.getopt(argv, "l:m:", ["label=", "max="])
    except getopt.GetoptError:
        print('python webcam.py --label Jokowo --max 5')
        sys.exit(2)

    is_training, label, max = False, '', 5
    for opt, arg in opts:
        if opt in ("-l", "--label"):
            is_training = True
            label = arg
        elif opt in ("-m", "--max"):
            max = arg

    return is_training, label, int(max)


def main(argv):
    is_training, label, max_training = read_args(argv)
    print(f'{is_training}|{label}|{max_training}')

    cam = cv2.VideoCapture(0)
    isShown = False
    training_count = 0

    while cam.isOpened():
        check, frame = cam.read()
        if check:
            # COLOR_BGR2GRAY |COLOR_BGR2RGB
            img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_cv2.detect_face(img_gray)
            print(f'{len(faces)} faces detected...')
            for face in faces:
                cv2.rectangle(frame, (face['x'], face['y']), (face['x'] +
                                                              face['w'], face['y'] + face['h']), (255, 0, 0), 3)

                face_crop = img_gray[face['y']:face['y'] +
                                     face['h'], face['x']:face['x'] + face['w']]
                label = face_tf.recognize(face_crop, is_training, label)

                cv2.putText(
                    frame, label, (face['x'], face['y'] - 10), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))

                if is_training:
                    training_count = training_count + 1
                    os.makedirs(f'temp/faces/{label}/', exist_ok=True)
                    cv2.imwrite(f'temp/faces/{label}/{label}_{time.time()}.png', face_crop)
                    break

                # if c % 50 == 0:
                #     print('---- SAVING -------')
                #     cv2.imwrite(f'face_{c}.png', face_crop)
            # face_tf.detect_face(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # if not isShown and c == 50:
            #     cv2.imshow('Image', img_gray)
            #     isShown = True

            cv2.imshow('video', frame)
            if is_training and training_count >= max_training:
                print('### TRAINING FINISH ###')
                break;

            key = cv2.waitKey(10)
            if key == 27:
                break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
