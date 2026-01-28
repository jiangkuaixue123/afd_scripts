// gcc -o check_rdma check_rdma.c -lrdmacm

// ./check_rdma

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <rdma/rdma_cma.h>

int main() {
    printf("--------------------------------------------------\n");
    printf("Starting RDMA Event Channel Check\n");
    printf("--------------------------------------------------\n");

    // 1. 尝试创建事件通道
    // 这通常是 RDMA 编程的第一步，它会打开 /dev/infiniband/rdma_cm
    struct rdma_event_channel *ec = rdma_create_event_channel();

    if (ec == NULL) {
        // 捕获并打印错误
        int err = errno;
        fprintf(stderr, "[FAILURE] rdma_create_event_channel returned NULL.\n");
        fprintf(stderr, "  Error Number (errno): %d\n", err);
        fprintf(stderr, "  Error Description   : %s\n", strerror(err));
        
        // 针对常见错误的提示
        if (err == ENOENT) { // Error 2
            fprintf(stderr, "\n[HINT] ENOENT usually means the kernel module is not loaded.\n");
            fprintf(stderr, "       Try: sudo modprobe rdma_cm\n");
            fprintf(stderr, "       Check if /dev/infiniband/rdma_cm exists.\n");
        } else if (err == EACCES) { // Error 13
            fprintf(stderr, "\n[HINT] EACCES means permission denied.\n");
            fprintf(stderr, "       Try running with sudo, or check ulimit -l (memlock).\n");
        }
        
        return 1;
    }

    printf("[SUCCESS] Event channel created successfully.\n");
    printf("  Channel File Descriptor: %d\n", ec->fd);

    // 2. 顺便检查一下能否获取到设备列表 (可选，但很有用)
    struct rdma_cm_id *id;
    // 创建一个临时的 ID 绑定到这个 channel，仅仅为���测试
    if (rdma_create_id(ec, &id, NULL, RDMA_PS_TCP) == 0) {
        printf("[SUCCESS] rdma_create_id also works (RDMA_PS_TCP).\n");
        rdma_destroy_id(id);
    } else {
        fprintf(stderr, "[WARNING] rdma_create_event_channel worked, but rdma_create_id failed: %s\n", strerror(errno));
    }

    // 3. 清理资源
    rdma_destroy_event_channel(ec);
    printf("--------------------------------------------------\n");
    printf("Check passed. RDMA CM environment seems OK.\n");
    
    return 0;
}
