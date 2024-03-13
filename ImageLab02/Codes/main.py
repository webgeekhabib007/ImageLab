import gaussian_filter
import mean_filter
import laplacian_filter
import log_filter
import sobel_filter
import grayscale_convolution
import RGB_convolution
def kernelSelector(n):
    while (1):
        print("Select a kernel : ")
        print("1. Gaussian kernel")
        print("2. Mean kernel")
        print("3. Laplacian kernel")
        print("4. LoG kernel")
        print("5. Sobel kernel")
        choice = int(input())
        if choice == 1:
            print("Enter SigmaX and SigmaY : ")
            values = input().split()    ## default value: 1 1
            values1, values2 = values
            sigmax = float(values1)
            sigmay = float(values2)
            # print(sigmax)
            # print(sigmay)

            gaussian_kernel = gaussian_filter.gaussian(sigmax=sigmax, sigmay=sigmay)
            # print(gaussian_kernel)
            return (gaussian_kernel,"gaussian")

        elif choice == 2:
            mean_kernel = mean_filter.mean(3)
            # print(mean_kernel)
            return (mean_kernel,"mean")
        elif choice == 3:
            while (1):
                print("Select a laplacian kernel")
                print("1. Center positive laplacian kernel")
                print("2. Center negative laplacian kernel")
                laplacian_positive_kernel, laplacian_negative_kernel = laplacian_filter.laplacian(3)
                k = int(input())
                if k == 1:
                    print("Laplacian positive kernel")
                    print(laplacian_positive_kernel)
                    return (laplacian_positive_kernel,"laplacian_positive")
                elif k == 2:
                    print("Laplacian negative kernel")
                    print(laplacian_negative_kernel)
                    return (laplacian_negative_kernel,"laplacian_negative")
                else:
                    print(f"Option {k} doesn't exist")
        elif choice==4:
            print("Enter the value of sigma")
            sigma=float(input())
            # print(sigma)
            log_kernel = log_filter.log(sigma=sigma)
            return (log_kernel,"log")
        elif choice==5:
            horizontal_kernel,vertical_kernel=sobel_filter.sobel()
            # print("Horizontal sobel filter")
            # print(horizontal_kernel)
            # print("Vertical sobel filter")
            # print(vertical_kernel)
            return (sobel_filter.sobel(),"sobel")
        else:
            print(f" {choice} : This is not a valid choice")

while(1):
    print("Select an option")
    print("1. Grayscale convolution")
    print("2. RGB and HSV convolution")
    choice=int(input())
    if choice==1:
        # Grayscale convolution
        kernel_with_type = kernelSelector(1)
        kernel,type=kernel_with_type
        print("kernel")
        if type == "sobel":
            horizontal_kernel, vertical_kernel = kernel
            print("Horizontal sobel filter")
            print(horizontal_kernel)
            print("Vertical sobel filter")
            print(vertical_kernel)

        else:
            print(kernel)
        print("Select the center : ")

        values = input().split()
        values1, values2 = values
        centerx = int(values1)
        centery = int(values2)
        # print(centerx)
        # print(centery)
        grayscale_convolution.grayscaleConvolution(kernel_with_type=kernel_with_type,center=(centerx,centery))


    elif choice==2:
        # RGB and HSV convolution
        kernel_with_type = kernelSelector(1)
        kernel, type = kernel_with_type
        print("kernel")
        if type == "sobel":
            horizontal_kernel, vertical_kernel = kernel
            print("Horizontal sobel filter")
            print(horizontal_kernel)
            print("Vertical sobel filter")
            print(vertical_kernel)

        else:
            print(kernel)
        print("Select the center")

        values = input().split()
        values1, values2 = values
        centerx = int(values1)
        centery = int(values2)
        # print(centerx)
        # print(centery)
        RGB_convolution.RGB_convolution(kernel_with_type=kernel_with_type, center=(centerx, centery))


    else:
        print(f" {choice} : This is not a valid choice")