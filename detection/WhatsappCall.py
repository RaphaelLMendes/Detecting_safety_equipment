import pywhatkit


# import time

class WhatappCall:
    '''class to make whatsapp msgs'''

    def SendWhatsApp(number, img_path,
                     msg='Olá Profissional de segurança do trabalho%2C%0AEsta mensagem é gerada automaticamente, então não precisa responder.%0A%0AFoi detectado a *falta de uso de EPI* na obra.'):
        # now = time.localtime()
        pywhatkit.sendwhats_image(number, img_path, msg)
        # if now.tm_min != 59:
        #     pywhatkit.sendwhatmsg(number,
        #                           msg,
        #                           now.tm_hour,
        #                           now.tm_min+1)
        #     pywhatkit.sendwhats_image(number, img_path, msg)
        #
        # else:
        #     pywhatkit.sendwhatmsg(number,
        #                           msg,
        #                           now.tm_hour+1,
        #                           0)


# WhatappCall.SendWhatsApp('+5511949314360')

if __name__ == "__main__":
    WhatappCall.SendWhatsApp('+5511949314360', 'images/screenshot_notsafe/3_W_hat_N_mask.mp4NOTSAFE0.jpg')
