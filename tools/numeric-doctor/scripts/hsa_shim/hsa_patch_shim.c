/* LD_PRELOAD shim that intercepts device code objects on their way to the GPU.
 *
 * WHY THIS EXISTS
 * ---------------
 * The mutation sweep (step 7) patches one kernel and replays.  Reaching the kernel
 * is backend-specific and getting steadily worse:
 *
 *   hipBLASLt/Tensile  unbundle a 43 MB fat binary, patch, rebundle, point
 *                      HIPBLASLT_TENSILE_LIBPATH at a shadow library dir
 *   aiter              patch a standalone .co, point AITER_ASM_DIR at a shadow tree
 *   CK                 ...device code is COMPILED INTO libtorch_hip.so, inside 5861
 *                      compressed (CCOB) offload bundles in a 94 MB .hip_fatbin
 *                      section.  There is no path to point anywhere.
 *
 * Patching CK the same way would mean parsing CCOB, zlib round-tripping a bundle that
 * decompresses 30x, recompressing it to fit the original span, and copying a 362 MB
 * library per overlay -- and it would still do nothing for Triton or whatever is next.
 *
 * So intercept lower down instead.  Every backend's device code reaches the GPU
 * through the HSA runtime, and HIP hands HSA the ALREADY-DECOMPRESSED ELF here:
 *
 *     libamdhip64.so:      U hsa_code_object_reader_create_from_memory@ROCR_1
 *     libhsa-runtime64.so: T hsa_code_object_reader_create_from_memory
 *
 * An undefined symbol resolved through the dynamic linker is interposable, so one
 * LD_PRELOAD object sees raw code objects for CK, aiter, Tensile and Triton alike.
 *
 * WHAT THIS DELIBERATELY DOES NOT DO
 * ----------------------------------
 * No ELF parsing, no disassembly, no knowledge of kernels.  It matches a byte string
 * and splices bytes at offsets the Python side computed.  Those offsets are exact by
 * construction because Python computes them from a DUMP taken through this same hook:
 * the bytes it analysed and the bytes patched here are the same bytes.
 *
 * THE LOG IS NOT OPTIONAL
 * -----------------------
 * The aiter work produced a sweep where the overlay was built, the kernel patched and
 * the environment delivered -- and the patch never reached the GPU, because the op had
 * dispatched to another backend entirely.  Every row still came back NAN and the
 * control still looked healthy.  A real sweep and a wholly fake one printed the SAME
 * table; only the loader's own trace could tell them apart.  NUMERIC_DOCTOR_HSA_LOG is
 * that trace for this path, and the sweep refuses to report rows it cannot find in it.
 *
 * MODES (env, so the Python side stays in control)
 *   NUMERIC_DOCTOR_HSA_DUMP=<dir>    write each code object seen, then call through
 *   NUMERIC_DOCTOR_HSA_MATCH=<str>   only act on objects containing this substring
 *   NUMERIC_DOCTOR_HSA_SPEC=<file>   apply patches to matching objects
 *   NUMERIC_DOCTOR_HSA_LOG=<file>    append one line per object seen
 *
 * SPEC FORMAT (line oriented; parsed with no allocator games, on purpose)
 *   MATCH <ascii-substring-no-spaces>
 *   PATCH <decimal-file-offset> <hex-bytes>      same-size splice
 *   FILE  <path>                                 replace the object wholesale
 *
 * FILE exists because some mutations INSERT instructions, which makes the code object
 * bigger, which means rewriting section and segment offsets.  Doing that in C, in the
 * loader's hot path, would be reckless; doing it in Python with the whole file in hand
 * is routine.  So the shim just hands HSA a different buffer of a different size --
 * which it can, because the API takes a pointer and a length.  FILE wins over PATCH if
 * both appear.
 *
 * BUILD: scripts/hsa_shim/build_hsa_shim.sh
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* The real prototype, from /opt/rocm/include/hsa/hsa.h.  Spelled out rather than
 * #include'd so the shim builds with a bare gcc and no ROCm headers on the include
 * path -- it is loaded into a process that already has the real runtime, so only the
 * ABI has to match, and this is it. */
typedef struct { uint64_t handle; } nd_code_object_reader_t;
typedef int nd_status_t;

#define ND_MAX_PATCHES 4096
#define ND_MAX_PATCH_BYTES 64

struct nd_patch {
    size_t off;
    unsigned char bytes[ND_MAX_PATCH_BYTES];
    size_t len;
};

static nd_status_t (*nd_real)(const void *, size_t, nd_code_object_reader_t *);
static int nd_ready;
static const char *nd_dump, *nd_match, *nd_spec, *nd_logp;
static struct nd_patch nd_patches[ND_MAX_PATCHES];
static size_t nd_npatches;
static int nd_spec_loaded;
static unsigned long nd_seen;
static char nd_replace[4096];           /* FILE <path>, empty if unset */

static void nd_log(const char *fmt, ...)
{
    if (!nd_logp)
        return;
    FILE *f = fopen(nd_logp, "a");
    if (!f)
        return;
    va_list ap;
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fclose(f);
}

static int nd_hexval(int c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

/* Reads MATCH/PATCH lines.  A malformed line is skipped and logged rather than
 * fatal: a trial that silently patched NOTHING must still be distinguishable from a
 * trial that patched something, and the log is what distinguishes them. */
static void nd_load_spec(void)
{
    nd_spec_loaded = 1;
    if (!nd_spec)
        return;
    FILE *f = fopen(nd_spec, "r");
    if (!f) {
        nd_log("spec-open-failed %s", nd_spec);
        return;
    }
    static char line[1 << 16];
    static char matchbuf[4096];
    while (fgets(line, sizeof(line), f)) {
        if (!strncmp(line, "MATCH ", 6)) {
            char *s = line + 6, *e = s + strlen(s);
            while (e > s && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' '))
                --e;
            size_t n = (size_t)(e - s);
            if (n && n < sizeof(matchbuf)) {
                memcpy(matchbuf, s, n);
                matchbuf[n] = 0;
                nd_match = matchbuf;
            }
        } else if (!strncmp(line, "FILE ", 5)) {
            char *s = line + 5, *e = s + strlen(s);
            while (e > s && (e[-1] == '\n' || e[-1] == '\r' || e[-1] == ' '))
                --e;
            size_t n = (size_t)(e - s);
            if (n && n < sizeof(nd_replace)) {
                memcpy(nd_replace, s, n);
                nd_replace[n] = 0;
            } else {
                nd_log("spec-bad-file %s", line);
            }
        } else if (!strncmp(line, "PATCH ", 6)) {
            if (nd_npatches >= ND_MAX_PATCHES) {
                nd_log("spec-too-many-patches");
                break;
            }
            char *s = line + 6;
            char *end = NULL;
            unsigned long long off = strtoull(s, &end, 10);
            if (end == s) {
                nd_log("spec-bad-offset %s", line);
                continue;
            }
            while (*end == ' ')
                ++end;
            struct nd_patch *p = &nd_patches[nd_npatches];
            p->off = (size_t)off;
            p->len = 0;
            while (end[0] && end[0] != '\n' && end[0] != '\r' && p->len < ND_MAX_PATCH_BYTES) {
                int hi = nd_hexval(end[0]), lo = nd_hexval(end[1]);
                if (hi < 0 || lo < 0)
                    break;
                p->bytes[p->len++] = (unsigned char)((hi << 4) | lo);
                end += 2;
            }
            if (p->len)
                ++nd_npatches;
            else
                nd_log("spec-bad-bytes %s", line);
        }
    }
    fclose(f);
    nd_log("spec-loaded patches=%zu match=%s", nd_npatches, nd_match ? nd_match : "(none)");
}

/* Finding the real function is not just dlsym(RTLD_NEXT).
 *
 * RTLD_NEXT searches the GLOBAL scope after this object, and libhsa-runtime64 is not
 * in it: HIP dlopen()s the runtime without RTLD_GLOBAL, so the only thing holding a
 * reference is libamdhip64's own local scope.  RTLD_NEXT therefore returns NULL even
 * though the call we are servicing came FROM that very library -- measured, not
 * theorised: the first build logged "FATAL dlsym(RTLD_NEXT) found no real symbol" on
 * every call.
 *
 * So ask for the runtime by name.  RTLD_NOLOAD first, to bind to the instance already
 * mapped rather than mapping a second copy of a runtime that owns GPU state. */
static void *nd_resolve(void)
{
    void *fn = dlsym(RTLD_NEXT, "hsa_code_object_reader_create_from_memory");
    if (fn)
        return fn;
    static const char *soname[] = {"libhsa-runtime64.so.1", "libhsa-runtime64.so"};
    for (size_t i = 0; i < sizeof(soname) / sizeof(soname[0]); ++i) {
        void *h = dlopen(soname[i], RTLD_LAZY | RTLD_NOLOAD);
        if (!h)
            h = dlopen(soname[i], RTLD_LAZY);
        if (!h)
            continue;
        fn = dlsym(h, "hsa_code_object_reader_create_from_memory");
        if (fn) {
            nd_log("resolved via dlopen(%s)", soname[i]);
            return fn;
        }
    }
    return NULL;
}

static void nd_init(void)
{
    if (nd_ready)
        return;
    nd_ready = 1;
    nd_dump = getenv("NUMERIC_DOCTOR_HSA_DUMP");
    nd_match = getenv("NUMERIC_DOCTOR_HSA_MATCH");
    nd_spec = getenv("NUMERIC_DOCTOR_HSA_SPEC");
    nd_logp = getenv("NUMERIC_DOCTOR_HSA_LOG");
    nd_real = nd_resolve();
    if (!nd_real)
        nd_log("FATAL could not resolve the real "
               "hsa_code_object_reader_create_from_memory");
    else
        nd_log("shim-ready pid=%d dump=%s spec=%s", (int)getpid(),
               nd_dump ? nd_dump : "-", nd_spec ? nd_spec : "-");
}

nd_status_t hsa_code_object_reader_create_from_memory(
    const void *code_object, size_t size, nd_code_object_reader_t *reader)
{
    nd_init();
    if (!nd_real) {
        /* Nothing sane to do: returning an error here would look like a GPU fault.
         * HSA_STATUS_ERROR == 0x1000. */
        return 0x1000;
    }
    if (!nd_spec_loaded)
        nd_load_spec();

    unsigned long n = ++nd_seen;
    int matched = 0;
    if (code_object && size && nd_match && *nd_match)
        matched = memmem(code_object, size, nd_match, strlen(nd_match)) != NULL;
    else if (!nd_match || !*nd_match)
        matched = 1;                      /* no filter: everything matches */

    if (nd_dump && matched) {
        char path[4096];
        snprintf(path, sizeof(path), "%s/co_%06lu_%zu.co", nd_dump, n, size);
        FILE *f = fopen(path, "wb");
        if (f) {
            fwrite(code_object, 1, size, f);
            fclose(f);
            nd_log("dumped n=%lu size=%zu -> %s", n, size, path);
        } else {
            nd_log("dump-failed n=%lu size=%zu path=%s", n, size, path);
        }
    }

    if (nd_spec && matched && nd_replace[0]) {
        /* Whole-object replacement: the Python side rebuilt this code object, very
         * likely at a DIFFERENT size (an inserted instruction grows .text and moves
         * every section after it).  Nothing here validates the ELF -- if it is
         * malformed HSA will reject it and the trial becomes an honest ERROR row,
         * which is the right outcome and far better than this shim guessing. */
        FILE *rf = fopen(nd_replace, "rb");
        if (!rf) {
            nd_log("replace-open-failed n=%lu path=%s -- PASSING THROUGH UNPATCHED",
                   n, nd_replace);
        } else {
            fseek(rf, 0, SEEK_END);
            long rsz = ftell(rf);
            fseek(rf, 0, SEEK_SET);
            unsigned char *buf = (rsz > 0) ? malloc((size_t)rsz) : NULL;
            if (buf && fread(buf, 1, (size_t)rsz, rf) == (size_t)rsz) {
                fclose(rf);
                /* Leaked on purpose; see the malloc note below. */
                nd_log("REPLACED n=%lu oldsize=%zu newsize=%ld applied=1 rejected=0",
                       n, size, rsz);
                return nd_real(buf, (size_t)rsz, reader);
            }
            fclose(rf);
            free(buf);
            nd_log("replace-read-failed n=%lu path=%s -- PASSING THROUGH UNPATCHED",
                   n, nd_replace);
        }
    }

    if (nd_spec && matched && nd_npatches) {
        /* Deliberately leaked.  HSA requires the buffer to stay valid until the
         * reader is destroyed, and a trial process loads a handful of patched
         * objects then exits -- freeing correctly would mean tracking reader
         * lifetimes for no benefit. */
        unsigned char *copy = malloc(size);
        if (!copy) {
            nd_log("alloc-failed n=%lu size=%zu -- PASSING THROUGH UNPATCHED", n, size);
        } else {
            memcpy(copy, code_object, size);
            size_t applied = 0, rejected = 0;
            for (size_t i = 0; i < nd_npatches; ++i) {
                struct nd_patch *p = &nd_patches[i];
                /* Bounds check before every splice.  A wrong offset here would
                 * corrupt an unrelated kernel in the same code object and the sweep
                 * would report nonsense about a kernel it never meant to touch. */
                if (p->off + p->len > size) {
                    ++rejected;
                    continue;
                }
                memcpy(copy + p->off, p->bytes, p->len);
                ++applied;
            }
            nd_log("PATCHED n=%lu size=%zu applied=%zu rejected=%zu", n, size,
                   applied, rejected);
            return nd_real(copy, size, reader);
        }
    }

    if (!nd_dump && !nd_spec)
        nd_log("seen n=%lu size=%zu matched=%d", n, size, matched);
    return nd_real(code_object, size, reader);
}
