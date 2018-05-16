import matplotlib.pyplot as plt
import random

class PrimitiveGUI(object):
    """
    A primitive terminal controlled image viewer GUI based on matplotlib
    Call from main to browse through images
    Images are chosen by an unsigned integer index
    GUI is left by entering "-1"
    """
    def __init__(self):
        # flag for gui control loop
        self.show = True
        # initial image


        # Target Array indices
        self.HIGH_LEVEL_COMMAND_IDX = 24 # int ( 2 Follow lane, 3 Left, 4 Right, 5 Straight)

        # make command verbose
        self.COMMAND_DICT =  {2: 'Follow Lane', 3: 'Left', 4: 'Right', 5: 'Straight'}

        # steering angle index from target vector
        self.STEER_IDX = 0 # float

    """ draws an arrow in current plot to indicate high level command """
    def draw_arrow(self):
        if self.verbose_cmd=="Straight":
            return plt.arrow(100,44,0,-10, width = 1, color='r')
        if self.verbose_cmd=="Follow Lane":
            return plt.arrow(100,44,0,-10, width = 1, color='r')
        if self.verbose_cmd=="Right":
            return plt.arrow(100,44,20,0, width = 1, color='r')
        if self.verbose_cmd=="Left":
            return plt.arrow(100,44,-20,0, width = 1, color='r')

    """ plot the images in one window """
    def plot_images(self, trans_img, orig_img, cmd, steering):
        # turn on interactive mode
        print(self.transformed_sample['filename'])
        print(self.idx%200)

        plt.ion()
        # create figure
        fig = plt.figure()

        # print command, steering angle, filename, index in plot title
        plt.suptitle("Command: {} | Steering Angle: {:.4f}\n \
        File: {} | Index: {}".format(cmd,steering,
        self.transformed_sample['filename'], self.idx%200))

        # add subfigure to show first image
        a=fig.add_subplot(1,2,1)
        a.set_title('Original')

        # draw arrow with high level command representation
        self.draw_arrow()

        # plot original picture with altered aspect ratio
        imgplot = plt.imshow(orig_img, aspect=3)

        # add subfigure to show second image
        a=fig.add_subplot(1,2,2)
        a.set_title('Transformed')

        # draw arrow with high level command representation
        self.draw_arrow()

        # plot transformed picture with altered aspect ratio
        imgplot = plt.imshow(trans_img, aspect=3)

        imgplot.set_clim(0.0,0.7)

        return fig

    """ communication loop via terminal """
    def ask_index(self):
        while True:
            self.verbose_cmd = self.COMMAND_DICT[
            self.transformed_sample['targets'][self.HIGH_LEVEL_COMMAND_IDX]
            ]

            self.steering_angle = self.transformed_sample['targets'][self.STEER_IDX]
            fig = self.plot_images(self.trans_img, self.orig_img,self.verbose_cmd,self.steering_angle)

            print("---------------------------------------------------------")
            print("Current Image file: {}| Image Index.: {}" \
            .format(self.transformed_sample['filename'], self.idx % 200))
            print("Command: {}| Steering Angle: {:.4f}".format(self.verbose_cmd,
                    self.steering_angle))
            print("---------------------------------------------------------")

            # only integers allowed, when value error, set raw to false
            try:
                raw = int(input('Enter new index [-1 to quit| Empty for random]: '))
            except ValueError:
                raw = False

            # check for errorous input
            if not raw:
                # new random index
                raw = random.randrange(0,self.len_train_set)

            # check input for exit flag ("-1")
            if raw>=0:
                self.idx = raw
                plt.close(fig)
                break

            else:
                self.show = False
                break

    def __call__(self, train_set, orig_train_set):
        # first image is randomly chosen
        self.idx = random.randrange(0,len(train_set))
        self.len_train_set = len(train_set)
        while self.show:

            # prepare transformed sample image
            self.transformed_sample = train_set[self.idx]
            self.trans_img = self.transformed_sample['data'].numpy().transpose((1,2,0))

            # prepare original sample image
            self.orig_sample = orig_train_set[self.idx]
            self.orig_img = self.orig_sample['data'].numpy().transpose((1,2,0))

            # ask for new index via terminal
            self.ask_index()
