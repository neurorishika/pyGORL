import numpy as np

class UnivalentDepression:
    # initialize the mushroom body
    def __init__(self, mu_inh=0.1, hr=0.9, lr=0.5, up_dr=5.0, fb_syn=0.1, fb_trans=0.1, fb_up=1.0):
        self.hr = hr  # homeostatic plasticity rate
        self.lr = lr  # learning rate
        self.eps = 1e-3  # small number to avoid division by zero
        self.w_KC_pMBON = np.array([1.0, 1.0])  # weights from KC to MBON (appetitive)
        self.w_KC_nMBON = np.array([1.0, 1.0])  # weights from KC to MBON (aversive)
        # mutual inhibition between MBONs (Felsenberg et al., 2018)
        self.w_nMBON_pMBON = 0.0 * mu_inh  # weight from aversive MBON to appetitive MBON
        self.w_pMBON_nMBON = 0.0 * mu_inh  # weight from appetitive MBON to aversive MBON
        # MBON to DAN feedback
        self.w_pMBON_pDANs = (
            0.0 * fb_syn
        )  # weight from appetitive MBON to reward DANs (inhibitory to subtract reward expectation)
        self.w_nMBON_nDANs = (
            0.0 * fb_syn
        )  # weight from aversive MBON to punishment DANs (inhibitory to subtract punishment expectation)
        self.w_pMBON_nDANs = (
            0.0 * fb_trans
        )  # weight from appetitive MBON to punishment DANs (excitatory to add reward expectation)
        self.w_nMBON_pDANs = (
            1.0 * fb_trans
        )  # weight from aversive MBON to reward DANs (excitatory to add punishment expectation)
        # Upwind Neuron inputs
        self.w_pMBON_U = (
            1.0 * up_dr
        )  # weight from appetitive MBON to upwind neuron (appetitive means upwind will be activated)
        self.w_nMBON_U = (
            -1.0 * up_dr
        )  # weight from aversive MBON to upwind neuron (aversive means upwind will be inhibited)
        # Upwind Neuron feedback to Dopamine Neurons
        self.w_U_pDANs = 1.0 * fb_up  # weight from upwind neuron to reward DANs
        self.w_U_nDANs = 0.0  # weight from upwind neuron to punishment DANs
        # Activation function
        # self.activation = lambda x: (1 / self.eps if x > 1 / self.eps else x) if x > 0 else 0  # ReLU
        self.activation = lambda x: np.clip(x, 0, 1 / self.eps)  # bounded ReLU

    # get the upwind drive for each odor without causing plasticity
    def upwind_drive(self):
        """
        A function to calculate the upwind drive for each odor without causing plasticity
        
        Parameters:
        -----------
        time_since_last_trial: float
            time since last trial in seconds (used for homeostatic plasticity, for now we set it to 1 arbitrary trial length)
        """
        drives = []
        for KC_activation in [np.array([1, 0]), np.array([0, 1])]:

            # Step 1: calculate the KC MBON weights after homeostatic plasticity (exponential decay back to 1)
            w_KC_pMBON_ = self.w_KC_pMBON + (1 - self.w_KC_pMBON) * (1 - np.exp(-self.hr))
            w_KC_nMBON_ = self.w_KC_nMBON + (1 - self.w_KC_nMBON) * (1 - np.exp(-self.hr))

            # Step 2: calculate the MBON activations
            MBON_activation = np.array(
                [
                    self.activation(np.dot(w_KC_pMBON_, KC_activation)),
                    self.activation(np.dot(w_KC_nMBON_, KC_activation)),
                ]
            )

            # Step 3: account for mutual inhibition between MBONs
            MBON_updated = np.array(
                [
                    self.activation(MBON_activation[0] + self.w_nMBON_pMBON * MBON_activation[1]),
                    self.activation(MBON_activation[1] + self.w_pMBON_nMBON * MBON_activation[0]),
                ]
            )

            # Step 4: calculate the upwind drive
            upwind_drive = self.activation(np.dot(MBON_updated, np.array([self.w_pMBON_U, self.w_nMBON_U])))
            drives.append(upwind_drive)

        return drives

    def trial_plasticity(self, odor, reward):
        """
        A function to calculate the plasticity after a trial
        
        Parameters:
        -----------
        odor: int
            odor 1 or odor 2
        reward: int
            reward or punishment
        time_since_last_trial: float
            time since last trial in seconds (used for homeostatic plasticity, for now we set it to 1 arbitrary trial length)
        """

        # Step 0: calculate the KC activations
        if odor == 0:
            KC_activation = np.array([1, 0])
        elif odor == 1:
            KC_activation = np.array([0, 1])

        # Step 0.5: calculate the DAN activations
        if reward == 1:
            pDAN_activation = 1
            nDAN_activation = 0
        elif reward == -1:
            pDAN_activation = 0
            nDAN_activation = 1
        else:
            pDAN_activation = 0
            nDAN_activation = 0

        # Step 1: calculate the KC MBON weights after homeostatic plasticity (exponential decay back to 1)
        self.w_KC_pMBON = self.w_KC_pMBON + (1 - self.w_KC_pMBON) * (1 - np.exp(-self.hr))
        self.w_KC_nMBON = self.w_KC_nMBON + (1 - self.w_KC_nMBON) * (1 - np.exp(-self.hr))

        # Step 2: calculate the MBON activations
        MBON_activation = np.array(
            [
                self.activation(np.dot(self.w_KC_pMBON, KC_activation)),
                self.activation(np.dot(self.w_KC_nMBON, KC_activation)),
            ]
        )

        # Step 3: account for mutual inhibition between MBONs
        MBON_updated = np.array(
            [
                self.activation(MBON_activation[0] + self.w_nMBON_pMBON * MBON_activation[1]),
                self.activation(MBON_activation[1] + self.w_pMBON_nMBON * MBON_activation[0]),
            ]
        )

        # Step 4: calculate the upwind drive
        upwind_drive = self.activation(np.dot(MBON_updated, np.array([self.w_pMBON_U, self.w_nMBON_U])))

        # Step 5: calculate the DAN activations
        pDAN_activation = self.activation(
            pDAN_activation
            + self.w_U_pDANs * upwind_drive
            + self.w_pMBON_pDANs * MBON_updated[0]
            - self.w_pMBON_pDANs  # to account for adaptation to typical DAN activation
            + self.w_nMBON_pDANs * MBON_updated[1]
            - self.w_nMBON_pDANs  # to account for adaptation to typical DAN activation
        )
        nDAN_activation = self.activation(
            nDAN_activation
            + self.w_U_nDANs * upwind_drive
            + self.w_pMBON_nDANs * MBON_updated[0]
            - self.w_pMBON_nDANs  # to account for adaptation to typical DAN activation
            + self.w_nMBON_nDANs * MBON_updated[1]
            - self.w_nMBON_nDANs  # to account for adaptation to typical DAN activation
        )

        # Step 6: calculate the plasticity and update the weights
        self.w_KC_pMBON = self.w_KC_pMBON - self.lr * nDAN_activation * KC_activation * self.w_KC_pMBON
        self.w_KC_nMBON = self.w_KC_nMBON - self.lr * pDAN_activation * KC_activation * self.w_KC_nMBON

        # Bound the weights
        self.w_KC_pMBON = np.clip(self.w_KC_pMBON, 0, 1)
        self.w_KC_nMBON = np.clip(self.w_KC_nMBON, 0, 1)

        # END of trial

    def get_weights(self):
        return self.w_KC_pMBON, self.w_KC_nMBON
            


    
class UnivalentBidirectional:
    # initialize the mushroom body
    def __init__(self, mu_inh=0.1, hr=0.9, lr=0.5, up_dr=5.0, fb_syn=0.1, fb_trans=0.1, fb_up=1.0):
        self.hr = hr  # homeostatic plasticity rate
        self.lr = lr  # learning rate
        self.eps = 1e-3  # small number to avoid division by zero
        self.w_KC_pMBON = np.array([1.0, 1.0])  # weights from KC to MBON (appetitive)
        self.w_KC_nMBON = np.array([1.0, 1.0])  # weights from KC to MBON (aversive)
        # mutual inhibition between MBONs (Felsenberg et al., 2018)
        self.w_nMBON_pMBON = 0.0 * mu_inh  # weight from aversive MBON to appetitive MBON
        self.w_pMBON_nMBON = 0.0 * mu_inh  # weight from appetitive MBON to aversive MBON
        # MBON to DAN feedback
        self.w_pMBON_pDANs = (
            0.0 * fb_syn
        )  # weight from appetitive MBON to reward DANs (inhibitory to subtract reward expectation)
        self.w_nMBON_nDANs = (
            0.0 * fb_syn
        )  # weight from aversive MBON to punishment DANs (inhibitory to subtract punishment expectation)
        self.w_pMBON_nDANs = (
            0.0 * fb_trans
        )  # weight from appetitive MBON to punishment DANs (excitatory to add reward expectation)
        self.w_nMBON_pDANs = (
            1.0 * fb_trans
        )  # weight from aversive MBON to reward DANs (excitatory to add punishment expectation)
        # Upwind Neuron inputs
        self.w_pMBON_U = (
            1.0 * up_dr
        )  # weight from appetitive MBON to upwind neuron (appetitive means upwind will be activated)
        self.w_nMBON_U = (
            -1.0 * up_dr
        )  # weight from aversive MBON to upwind neuron (aversive means upwind will be inhibited)
        # Upwind Neuron feedback to Dopamine Neurons
        self.w_U_pDANs = 1.0 * fb_up  # weight from upwind neuron to reward DANs
        self.w_U_nDANs = 0.0  # weight from upwind neuron to punishment DANs
        # Activation function
        # self.activation = lambda x: (1 / self.eps if x > 1 / self.eps else x) if x > 0 else 0  # ReLU
        self.activation = lambda x: np.clip(x, 0, 1 / self.eps)  # bounded ReLU
        self.bidirectional_activation = lambda x: np.clip(x, -1 / self.eps, 1 / self.eps)  # bounded ReLU

    # get the upwind drive for each odor without causing plasticity
    def upwind_drive(self):
        """
        A function to calculate the upwind drive for each odor without causing plasticity
        
        Parameters:
        -----------
        time_since_last_trial: float
            time since last trial in seconds (used for homeostatic plasticity, for now we set it to 1 arbitrary trial length)
        """
        drives = []
        for KC_activation in [np.array([1, 0]), np.array([0, 1])]:

            # Step 1: calculate the KC MBON weights after homeostatic plasticity (exponential decay back to 1)
            w_KC_pMBON_ = self.w_KC_pMBON + (1 - self.w_KC_pMBON) * (1 - np.exp(-self.hr))
            w_KC_nMBON_ = self.w_KC_nMBON + (1 - self.w_KC_nMBON) * (1 - np.exp(-self.hr))

            # Step 2: calculate the MBON activations
            MBON_activation = np.array(
                [
                    self.activation(np.dot(w_KC_pMBON_, KC_activation)),
                    self.activation(np.dot(w_KC_nMBON_, KC_activation)),
                ]
            )

            # Step 3: account for mutual inhibition between MBONs
            MBON_updated = np.array(
                [
                    self.activation(MBON_activation[0] + self.w_nMBON_pMBON * MBON_activation[1]),
                    self.activation(MBON_activation[1] + self.w_pMBON_nMBON * MBON_activation[0]),
                ]
            )

            # Step 4: calculate the upwind drive
            upwind_drive = self.activation(np.dot(MBON_updated, np.array([self.w_pMBON_U, self.w_nMBON_U])))
            drives.append(upwind_drive)

        return drives

    def trial_plasticity(self, odor, reward):
        """
        A function to calculate the plasticity after a trial
        
        Parameters:
        -----------
        odor: int
            odor 1 or odor 2
        reward: int
            reward or punishment
        time_since_last_trial: float
            time since last trial in seconds (used for homeostatic plasticity, for now we set it to 1 arbitrary trial length)
        """

        # Step 0: calculate the KC activations
        if odor == 0:
            KC_activation = np.array([1, 0])
        elif odor == 1:
            KC_activation = np.array([0, 1])

        # Step 0.5: calculate the DAN activations
        if reward == 1:
            pDAN_activation = 1
            nDAN_activation = 0
        elif reward == -1:
            pDAN_activation = 0
            nDAN_activation = 1
        else:
            pDAN_activation = 0
            nDAN_activation = 0

        # Step 1: calculate the KC MBON weights after homeostatic plasticity (exponential decay back to 1)
        self.w_KC_pMBON = self.w_KC_pMBON + (1 - self.w_KC_pMBON) * (1 - np.exp(-self.hr))
        self.w_KC_nMBON = self.w_KC_nMBON + (1 - self.w_KC_nMBON) * (1 - np.exp(-self.hr))

        # Step 2: calculate the MBON activations
        MBON_activation = np.array(
            [
                self.activation(np.dot(self.w_KC_pMBON, KC_activation)),
                self.activation(np.dot(self.w_KC_nMBON, KC_activation)),
            ]
        )

        # Step 3: account for mutual inhibition between MBONs
        MBON_updated = np.array(
            [
                self.activation(MBON_activation[0] + self.w_nMBON_pMBON * MBON_activation[1]),
                self.activation(MBON_activation[1] + self.w_pMBON_nMBON * MBON_activation[0]),
            ]
        )

        # Step 4: calculate the upwind drive
        upwind_drive = self.activation(np.dot(MBON_updated, np.array([self.w_pMBON_U, self.w_nMBON_U])))

        # Step 5: calculate the DAN activations
        pDAN_activation = self.bidirectional_activation(
            pDAN_activation
            + self.w_U_pDANs * upwind_drive
            + self.w_pMBON_pDANs * MBON_updated[0]
            - self.w_pMBON_pDANs  # to account for adaptation to typical DAN activation
            + self.w_nMBON_pDANs * MBON_updated[1]
            - self.w_nMBON_pDANs  # to account for adaptation to typical DAN activation
        )
        nDAN_activation = self.bidirectional_activation(
            nDAN_activation
            + self.w_U_nDANs * upwind_drive
            + self.w_pMBON_nDANs * MBON_updated[0]
            - self.w_pMBON_nDANs  # to account for adaptation to typical DAN activation
            + self.w_nMBON_nDANs * MBON_updated[1]
            - self.w_nMBON_nDANs  # to account for adaptation to typical DAN activation
        )

        # Step 6: calculate the plasticity and update the weights
        self.w_KC_pMBON = self.w_KC_pMBON - self.lr * nDAN_activation * KC_activation * self.w_KC_pMBON
        self.w_KC_nMBON = self.w_KC_nMBON - self.lr * pDAN_activation * KC_activation * self.w_KC_nMBON

        # Bound the weights
        self.w_KC_pMBON = np.clip(self.w_KC_pMBON, 0, 1)
        self.w_KC_nMBON = np.clip(self.w_KC_nMBON, 0, 1)

        # END of trial

    def get_weights(self):
        return self.w_KC_pMBON, self.w_KC_nMBON
    
    
class BivalentDepression:
    # initialize the mushroom body
    def __init__(self, mu_inh=0.1, hr=0.9, lr=0.5, up_dr=5.0, fb_syn=0.1, fb_trans=0.1, fb_up=1.0):
        self.hr = hr  # homeostatic plasticity rate
        self.lr = lr  # learning rate
        self.eps = 1e-3  # small number to avoid division by zero
        self.w_KC_pMBON = np.array([1.0, 1.0])  # weights from KC to MBON (appetitive)
        self.w_KC_nMBON = np.array([1.0, 1.0])  # weights from KC to MBON (aversive)
        # mutual inhibition between MBONs (Felsenberg et al., 2018)
        self.w_nMBON_pMBON = -1.0 * mu_inh  # weight from aversive MBON to appetitive MBON
        self.w_pMBON_nMBON = -1.0 * mu_inh  # weight from appetitive MBON to aversive MBON
        # MBON to DAN feedback
        self.w_pMBON_pDANs = (
            -1.0 * fb_syn
        )  # weight from appetitive MBON to reward DANs (inhibitory to subtract reward expectation)
        self.w_nMBON_nDANs = (
            -1.0 * fb_syn
        )  # weight from aversive MBON to punishment DANs (inhibitory to subtract punishment expectation)
        self.w_pMBON_nDANs = (
            1.0 * fb_trans
        )  # weight from appetitive MBON to punishment DANs (excitatory to add reward expectation)
        self.w_nMBON_pDANs = (
            1.0 * fb_trans
        )  # weight from aversive MBON to reward DANs (excitatory to add punishment expectation)
        # Upwind Neuron inputs
        self.w_pMBON_U = (
            1.0 * up_dr
        )  # weight from appetitive MBON to upwind neuron (appetitive means upwind will be activated)
        self.w_nMBON_U = (
            -1.0 * up_dr
        )  # weight from aversive MBON to upwind neuron (aversive means upwind will be inhibited)
        # Upwind Neuron feedback to Dopamine Neurons
        self.w_U_pDANs = 1.0 * fb_up  # weight from upwind neuron to reward DANs
        self.w_U_nDANs = 0.0  # weight from upwind neuron to punishment DANs
        # Activation function
        # self.activation = lambda x: (1 / self.eps if x > 1 / self.eps else x) if x > 0 else 0  # ReLU
        self.activation = lambda x: np.clip(x, 0, 1 / self.eps)  # bounded ReLU

    # get the upwind drive for each odor without causing plasticity
    def upwind_drive(self):
        """
        A function to calculate the upwind drive for each odor without causing plasticity
        
        Parameters:
        -----------
        time_since_last_trial: float
            time since last trial in seconds (used for homeostatic plasticity, for now we set it to 1 arbitrary trial length)
        """
        drives = []
        for KC_activation in [np.array([1, 0]), np.array([0, 1])]:

            # Step 1: calculate the KC MBON weights after homeostatic plasticity (exponential decay back to 1)
            w_KC_pMBON_ = self.w_KC_pMBON + (1 - self.w_KC_pMBON) * (1 - np.exp(-self.hr))
            w_KC_nMBON_ = self.w_KC_nMBON + (1 - self.w_KC_nMBON) * (1 - np.exp(-self.hr))

            # Step 2: calculate the MBON activations
            MBON_activation = np.array(
                [
                    self.activation(np.dot(w_KC_pMBON_, KC_activation)),
                    self.activation(np.dot(w_KC_nMBON_, KC_activation)),
                ]
            )

            # Step 3: account for mutual inhibition between MBONs
            MBON_updated = np.array(
                [
                    self.activation(MBON_activation[0] + self.w_nMBON_pMBON * MBON_activation[1]),
                    self.activation(MBON_activation[1] + self.w_pMBON_nMBON * MBON_activation[0]),
                ]
            )

            # Step 4: calculate the upwind drive
            upwind_drive = self.activation(np.dot(MBON_updated, np.array([self.w_pMBON_U, self.w_nMBON_U])))
            drives.append(upwind_drive)

        return drives

    def trial_plasticity(self, odor, reward):
        """
        A function to calculate the plasticity after a trial
        
        Parameters:
        -----------
        odor: int
            odor 1 or odor 2
        reward: int
            reward or punishment
        time_since_last_trial: float
            time since last trial in seconds (used for homeostatic plasticity, for now we set it to 1 arbitrary trial length)
        """

        # Step 0: calculate the KC activations
        if odor == 0:
            KC_activation = np.array([1, 0])
        elif odor == 1:
            KC_activation = np.array([0, 1])

        # Step 0.5: calculate the DAN activations
        if reward == 1:
            pDAN_activation = 1
            nDAN_activation = 0
        elif reward == -1:
            pDAN_activation = 0
            nDAN_activation = 1
        else:
            pDAN_activation = 0
            nDAN_activation = 0

        # Step 1: calculate the KC MBON weights after homeostatic plasticity (exponential decay back to 1)
        self.w_KC_pMBON = self.w_KC_pMBON + (1 - self.w_KC_pMBON) * (1 - np.exp(-self.hr))
        self.w_KC_nMBON = self.w_KC_nMBON + (1 - self.w_KC_nMBON) * (1 - np.exp(-self.hr))

        # Step 2: calculate the MBON activations
        MBON_activation = np.array(
            [
                self.activation(np.dot(self.w_KC_pMBON, KC_activation)),
                self.activation(np.dot(self.w_KC_nMBON, KC_activation)),
            ]
        )

        # Step 3: account for mutual inhibition between MBONs
        MBON_updated = np.array(
            [
                self.activation(MBON_activation[0] + self.w_nMBON_pMBON * MBON_activation[1]),
                self.activation(MBON_activation[1] + self.w_pMBON_nMBON * MBON_activation[0]),
            ]
        )

        # Step 4: calculate the upwind drive
        upwind_drive = self.activation(np.dot(MBON_updated, np.array([self.w_pMBON_U, self.w_nMBON_U])))

        # Step 5: calculate the DAN activations
        pDAN_activation = self.activation(
            pDAN_activation
            + self.w_U_pDANs * upwind_drive
            + self.w_pMBON_pDANs * MBON_updated[0]
            - self.w_pMBON_pDANs  # to account for adaptation to typical DAN activation
            + self.w_nMBON_pDANs * MBON_updated[1]
            - self.w_nMBON_pDANs  # to account for adaptation to typical DAN activation
        )
        nDAN_activation = self.activation(
            nDAN_activation
            + self.w_U_nDANs * upwind_drive
            + self.w_pMBON_nDANs * MBON_updated[0]
            - self.w_pMBON_nDANs  # to account for adaptation to typical DAN activation
            + self.w_nMBON_nDANs * MBON_updated[1]
            - self.w_nMBON_nDANs  # to account for adaptation to typical DAN activation
        )

        # Step 6: calculate the plasticity and update the weights
        self.w_KC_pMBON = self.w_KC_pMBON - self.lr * nDAN_activation * KC_activation * self.w_KC_pMBON
        self.w_KC_nMBON = self.w_KC_nMBON - self.lr * pDAN_activation * KC_activation * self.w_KC_nMBON

        # Bound the weights
        self.w_KC_pMBON = np.clip(self.w_KC_pMBON, 0, 1)
        self.w_KC_nMBON = np.clip(self.w_KC_nMBON, 0, 1)

        # END of trial

    def get_weights(self):
        return self.w_KC_pMBON, self.w_KC_nMBON