import 'vue';

declare module 'vue' {
    interface ComponentCustomProperties {
        /** PlayHex's templates call $t; main.ts binds i18next's own t to it. */
        $t: typeof import('i18next').t;
    }
}
