import numpy as np
from matplotlib import  pyplot as plt
import cv2
import os
import sys
from PIL import  Image
import pydicom
import tifffile as tif




def  tiftopng(path,new_path):
    '''
    :param path:
    :return:
    '''
    filesnames=os.listdir(path)
    for i in range(len(filesnames)):
        filename=filesnames[i]
        old_path=os.path.join(path,filename)
        image = tif.imread(old_path)
        image = Image.fromarray(image).convert('RGB')
        new_name=filename.split('.')[0]+".png"
        new_path_file=os.path.join(new_path,new_name)
        image.save(new_path_file)




def show_images(images,masks,o1,o2,filename):
    image = Image.fromarray(images)
    mask = Image.fromarray(masks)

    unet = Image.fromarray(o1)
    nas_search = Image.fromarray(o2)

    #
    # image.save(os.path.join(os.path.join(res_dir,'images'), '{}.png'.format(filename)))
    # mask.save(os.path.join(os.path.join(res_dir,'mask'), '{}.png'.format(filename)))
    # unet.save(os.path.join(os.path.join(res_dir,'unet'), '{}.png'.format(filename)))
    # nas_search.save(os.path.join(os.path.join(res_dir,'nas_search'), '{}.png'.format(filename)))



def main():

    # # cvc dataset
    # nas_better=[116,158,164,174,220,249,250,251,266,267,270,298,301,308,314,322,335,337,338,340,350,370,404,413,415,437,447,
    #             457,505,508,515,526,528,545,547/2,548/2,572,576,579,602,]
    # all_good=[98,128,130,149,180,279,290,296,319,488,504,]
    #
    # bad=[475,481,489,497,544,600,]
    #
    # #
    #
    # image_path=r'C:\Users\rileyliu\Desktop\CVC-ClinicDB\valid'
    # mask_path=r'C:\Users\rileyliu\Desktop\images_res\cvc\mask'
    # v1_path=r'C:\Users\rileyliu\Desktop\images_res\cvc\layer7_double_deep\images'
    # v3_path=r'C:\Users\rileyliu\Desktop\images_res\cvc\stage1_nodouble_deep\images'
    # #unet_path=r'C:\Users\rileyliu\Desktop\images_res\eval\unet\images'
    # alpha05_dd_path=r'C:\Users\rileyliu\Desktop\images_res\cvc\alpha0_5_double_deep\images'
    # unet_path=r'C:\Users\rileyliu\Desktop\images_res\cvc\unet_ep800dice\images'
    # output_dir=r'C:\Users\rileyliu\Desktop\images_res\cvc\compare_alpha05'
    #
    # #tiftopng(image_path,new_images_path)
    # # read
    # all_files=os.listdir(image_path)
    # for i in range(len(all_files)):
    #     filename=all_files[i]
    #     print("Inference {}".format(filename))
    #     fileindex=filename.split('.')[0]
    #     image_name=os.path.join(image_path,filename)
    #     mask_name=os.path.join(mask_path,fileindex+".tif")
    #     unet_file_name=os.path.join(unet_path,fileindex+".png")
    #
    #     # v1_file_name=os.path.join(v1_path,fileindex+".png")
    #     # v3_file_name=os.path.join(v3_path,fileindex+".png")
    #     alpha05_dd_file=os.path.join(alpha05_dd_path,fileindex+"_8.png")
    #
    #     # read mask
    #     image = tif.imread(image_name)
    #     image = Image.fromarray(image).convert('RGB')
    #     image = image.resize((256, 192), Image.BILINEAR)
    #
    #     mask=Image.open(mask_name)
    #     mask = mask.resize((256, 192), Image.NEAREST)
    #     unet=Image.open(unet_file_name)
    #
    #     # v1 = Image.open(v1_file_name)
    #     # v3=Image.open(v3_file_name)
    #     alpha05_dd=Image.open(alpha05_dd_file)
    #
    #     #
    #     image=np.asarray(image).astype(np.uint8)
    #     mask=np.asarray(mask).astype(np.uint8)
    #     mask=(mask==255).astype(np.uint8)
    #     unet=np.asarray(unet).astype(np.uint8)
    #     unet[unet>0]=255
    #     # v1=np.asarray(v1).astype(np.uint8)
    #     # v1[v1>0]=255
    #     #
    #     # v3=np.asarray(v3).astype(np.uint8)
    #     # v3[v3>0]=255
    #
    #     alpha05_dd=np.asarray(alpha05_dd).astype(np.uint8)
    #     alpha05_dd[alpha05_dd>0]=255
    #
    #     #rgb
    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(image, contours, -1, (0, 0,255), 1,lineType=cv2.LINE_AA)
    #     output_contours, _ = cv2.findContours(unet, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(image, output_contours, -1, (0, 255,0), 1,lineType=cv2.LINE_AA)
    #
    #     # output_contours, _ = cv2.findContours(v1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     # cv2.drawContours(image, output_contours, -1, (255, 0,0), 1,lineType=cv2.LINE_AA)
    #     #
    #     # output_contours, _ = cv2.findContours(v3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     # cv2.drawContours(image, output_contours, -1, (255, 255,0), 1,lineType=cv2.LINE_AA)
    #
    #     output_contours, _ = cv2.findContours(alpha05_dd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(image, output_contours, -1, (255, 0,0), 1,lineType=cv2.LINE_AA)
    #
    #     image = Image.fromarray(image)
    #     image.save(os.path.join(output_dir, '{}.png'.format(fileindex)))
    #



    # # isic2018 dataset
    # image_path=r'C:\Users\rileyliu\Desktop\ISIC2018\Task1_apply\valid'   # ISIC_.....jpg
    # mask_path=r'C:\Users\rileyliu\Desktop\images_res\isic2018\valid_GT'     #  ISIC_0000008_segmentation
    # #unet_path=r'C:\Users\rileyliu\Desktop\images_res\eval\unet\images'
    # unet_path=r'C:\Users\rileyliu\Desktop\images_res\isic2018_v2\unet\images'
    # dd_path=r'C:\Users\rileyliu\Desktop\images_res\isic2018_v2\double_deep\images'
    # alpha_0_5_dd_path=r'C:\Users\rileyliu\Desktop\images_res\isic2018_v2\alpha0_5_double_deep_0.01\images'
    # #slim_nodouble_deep_path=r'C:\Users\rileyliu\Desktop\images_res\isic2018_v2\slim_nodouble_deep\images'
    # slim_nodouble_deep_path=r'C:\Users\rileyliu\Desktop\images_res\isic2018_v2\nodouble_deep\images'
    # output_dir=r'C:\Users\rileyliu\Desktop\images_res\isic2018_v2\compare_alpha0_5_dd'
    # #original_images_path=r'C:\Users\rileyliu\Desktop\images_res\isic2018_v2\original_images'
    # #tiftopng(image_path,new_images_path)
    #
    # # read
    # all_files=os.listdir(image_path)
    # for i in range(len(all_files)):
    #     file_name=all_files[i]
    #     file_index=file_name.split('_')[-1]
    #     print(file_index)
    #     file_index=file_index.split('.')[0]
    #     print("Inference {}".format(file_index))
    #     maskname = 'ISIC_' + file_index + '_segmentation.png'
    #     image_name=os.path.join(image_path,file_name)
    #     mask_name=os.path.join(mask_path,maskname)
    #     unet_file_name=os.path.join(unet_path,maskname)
    #
    #     # dd_file_name=os.path.join(dd_path,maskname)
    #     # slim_nodouble_deep_file_name = os.path.join(slim_nodouble_deep_path, maskname)
    #     alpha_0_5_dd=os.path.join(alpha_0_5_dd_path,'ISIC_' + file_index + '_segmentation_8.png')
    #
    #     # read mask
    #     image = Image.open(image_name).convert('RGB')
    #     image = image.resize((256, 256), Image.BILINEAR)
    #     mask=Image.open(mask_name)
    #     mask = mask.resize((256, 256), Image.NEAREST)
    #
    #
    #     unet=Image.open(unet_file_name)
    #     # dd = Image.open(dd_file_name)
    #     # slim_nodouble_deep = Image.open(slim_nodouble_deep_file_name)
    #     alpha0_5_dd=Image.open(alpha_0_5_dd)
    #
    #
    #     #
    #     image=np.asarray(image).astype(np.uint8)
    #     mask=np.asarray(mask).astype(np.uint8)
    #     mask=(mask==255).astype(np.uint8)
    #     unet=np.asarray(unet).astype(np.uint8)
    #     unet[unet>0]=255
    #     # dd=np.asarray(dd).astype(np.uint8)
    #     # dd[dd>0]=255
    #     # slim_nodouble_deep=np.asarray(slim_nodouble_deep).astype(np.uint8)
    #     # slim_nodouble_deep[slim_nodouble_deep>0]=255
    #
    #     alpha0_5_dd=np.asarray(alpha0_5_dd).astype(np.uint8)
    #     alpha0_5_dd[alpha0_5_dd>0]=255
    #
    #
    #     #rgb
    #     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(image, contours, -1, (0, 0,255), 1,lineType=cv2.LINE_AA)
    #     output_contours, _ = cv2.findContours(unet, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(image, output_contours, -1, (0, 255,0), 1,lineType=cv2.LINE_AA)
    #
    #     output_contours, _ = cv2.findContours(alpha0_5_dd, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     cv2.drawContours(image, output_contours, -1, (255, 0,0), 1,lineType=cv2.LINE_AA)
    #
    #
    #     # output_contours, _ = cv2.findContours(slim_nodouble_deep, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     # cv2.drawContours(image, output_contours, -1, (255,255,0), 1,lineType=cv2.LINE_AA)
    #
    #     image = Image.fromarray(image)
    #     image.save(os.path.join(output_dir, file_name))
    #

    # chao dataset
    subdir_list=['14']
    image_path=r'C:\Users\rileyliu\Desktop\CHAO\valid'
    mask=r'C:\Users\rileyliu\Desktop\images_res\chaos_v1\mask'


    unet=r'C:\Users\rileyliu\Desktop\images_res\chaos_v1\unet\images'
    # unetpp=r'C:\Users\rileyliu\Desktop\images_res\chaos_v1\unet++\images'
    # multires_unet=r'C:\Users\rileyliu\Desktop\images_res\chaos_v1\multires_unet\images'
    # att_unet=r'C:\Users\rileyliu\Desktop\images_res\chaos_v1\attention_unet\images'
    # nodouble_deep=r'C:\Users\rileyliu\Desktop\images_res\chaos_v1\stage1_nodouble_deep_150awb_init32\images'
    # output_dir=r'C:\Users\rileyliu\Desktop\images_res\chaos_v1\compare'
    alpha05_dir=r'E:\segmentation\Image_Segmentation\nas_search_unet\eval\chaos\alpha0_5_double_deep\images'
    new_compare_dir=r'C:\Users\rileyliu\Desktop\images_res\chaos_v1\new_compare'
    #tiftopng(image_path,new_images_path)


    for i in range(len(subdir_list)):
        subdir_images_path=os.path.join(image_path,subdir_list[i],"DICOM_anon")
        masks_path=os.path.join(mask,subdir_list[i],'Ground')
        unet_path=os.path.join(unet,subdir_list[i])
        # unetpp_path=os.path.join(unetpp,subdir_list[i])
        # multires_unet_path=os.path.join(multires_unet,subdir_list[i])
        # att_unet_path=os.path.join(att_unet,subdir_list[i])
        # nodouble_deep_path=os.path.join(nodouble_deep,subdir_list[i])
        alpha05_path=os.path.join(alpha05_dir,subdir_list[i])


        images=os.listdir(subdir_images_path)
        for image_name in images:
            if 'IMG' in image_name:
                image_index=int(image_name[:-4].split('-')[-1][2:])-1
                image_index=str(image_index).rjust(3,'0')
                image_mask_name = 'liver_GT_' + str(image_index) + '.png'
                alpha05_index='liver_GT_' + str(image_index) + '_8.png'
            else:
                image_mask_name = 'liver_GT_' + image_name[:-4].split(',')[0][2:] + '.png'
                alpha05_index= 'liver_GT_' + image_name[:-4].split(',')[0][2:] + '_8.png'
            img_path = os.path.join(subdir_images_path, image_name)
            img_mask_path = os.path.join(masks_path, image_mask_name)
            img_unet_path = os.path.join(unet_path, image_mask_name)
            # img_unetpp_path = os.path.join(unetpp_path, image_mask_name)
            # img_multires_unet_path = os.path.join(multires_unet_path, image_mask_name)
            # img_att_unet_path = os.path.join(att_unet_path, image_mask_name)
            # img_nodouble_deep_path = os.path.join(nodouble_deep_path, image_mask_name)
            img_alpha05_path = os.path.join(alpha05_path, alpha05_index)


            print(img_path)
            img = pydicom.dcmread(img_path,force=True)
            #v=m*array+b
            img, itercept = img.RescaleSlope * img.pixel_array + img.RescaleIntercept, img.RescaleIntercept
            img[img >= 4000] = itercept  # C

            # import matplotlib.pyplot as plt
            # plt.imshow(img,'gray')
            # plt.show()


            # mask
            mask = Image.open(img_mask_path)
            mask = mask.resize((256,256), Image.NEAREST)
            mask=np.asarray(mask).astype(np.uint8)
            mask[mask>0]=255

            #unet
            unet_ = Image.open(img_unet_path)
            unet_=np.asarray(unet_).astype(np.uint8)
            unet_[unet_>0]=255

            # att unet
            img_alpha05 = Image.open(img_alpha05_path)
            img_alpha05=np.asarray(img_alpha05).astype(np.uint8)
            img_alpha05[img_alpha05>0]=255

            # # nodouble deep
            # nodouble_deep = Image.open(img_nodouble_deep_path)
            # nodouble_deep=np.asarray(nodouble_deep).astype(np.uint8)
            # nodouble_deep[nodouble_deep>0]=255


            # 512*512
            # image_data=img.pixel_array
            # h,w=image_data.shape
            # new_image=np.zeros(shape=(h,w,3),dtype=np.int32)
            # new_image[:,:,0]=image_data
            # new_image[:,:,1]=image_data
            # new_image[:,:,2]=image_data

            # scale to 0~255
            img = Image.fromarray(img).convert('L')
            img = img.resize((256, 256), Image.BILINEAR)
            img=np.asarray(img)
            h,w=img.shape
            new_image=np.zeros(shape=(h,w,3),dtype=np.uint8)
            new_image[:,:,0]=img
            new_image[:,:,1]=img
            new_image[:,:,2]=img



            # rgb
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(new_image, contours, -1, (0, 0,255), 1,lineType=cv2.LINE_AA)


            output_contours, _ = cv2.findContours(unet_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(new_image, output_contours, -1, (0, 255,0), 1,lineType=cv2.LINE_AA)

            # output_contours, _ = cv2.findContours(att_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.drawContours(new_image, output_contours, -1, (255, 255,0), 1,lineType=cv2.LINE_AA)

            output_contours, _ = cv2.findContours(img_alpha05, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(new_image, output_contours, -1, (255, 0,0), 1,lineType=cv2.LINE_AA)

            # import matplotlib.pyplot as plt
            # plt.imshow(new_image,'gray')
            # plt.show()

            # cv2.imshow('result', new_image)
            # cv2.waitKey()
            # image.save(os.path.join(new_compare_dir, file_name))

            image = Image.fromarray(new_image)
            image.save(os.path.join(new_compare_dir, image_mask_name))

# def plot_final(path):


if __name__=="__main__":
    main()


