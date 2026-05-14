import re

with open('e:/aitrade/frontend/src/components/Sidebar.tsx', 'r', encoding='utf-8') as f:
    code = f.read()

# Import User from lucide-react
if 'User' not in code[:500]:
    code = code.replace("ShieldCheck\n} from 'lucide-react';", "ShieldCheck, User\n} from 'lucide-react';")
    code = code.replace("ShieldCheck } from 'lucide-react';", "ShieldCheck, User } from 'lucide-react';")

# The new HTML block
new_block = '''        {/* User Management Section */}
        <div className="mb-1">
          {sidebarOpen && (
            <div className="flex items-center gap-2 px-2 pt-4 pb-1.5">
              <span className="w-1 h-1 rounded-full bg-[var(--border)]" />
              <span className="text-[9px] font-bold uppercase tracking-[0.15em] text-[var(--text-dim)]">
                User Management
              </span>
            </div>
          )}
          {!sidebarOpen && (
            <div className="mx-2 my-3 h-px bg-[var(--border-light)]" />
          )}
          <div className="space-y-0.5">
            {userRole === 'super_admin' ? (
              <>
                <NavLink
                  to="/admin?tab=roles"
                  className={({ isActive }) => `
                    flex items-center gap-3 px-3 py-2 rounded-[var(--radius-sm)]
                    text-[13px] font-medium transition-all duration-150 group relative
                    ${(isActive && location.search.includes('tab=roles'))
                      ? 'bg-[var(--primary-subtle)] text-[var(--primary)] border-l-2 border-[var(--primary)] pl-[10px]'
                      : 'text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text)]'
                    }
                  `}
                >
                  <ShieldCheck size={18} className={`flex-shrink-0 transition-opacity ${location.search.includes('tab=roles') ? '' : 'opacity-60 group-hover:opacity-100'}`} />
                  {sidebarOpen && <span className="truncate">Roles</span>}
                  {!sidebarOpen && (
                    <div className="
                      absolute left-full ml-3 px-3 py-1.5 rounded-md
                      bg-[var(--bg-elevated)] border border-[var(--border)]
                      text-xs text-[var(--text)] whitespace-nowrap
                      opacity-0 pointer-events-none group-hover:opacity-100
                      transition-opacity duration-150 shadow-[var(--shadow-lg)] z-50
                    ">
                      Roles
                    </div>
                  )}
                </NavLink>

                <NavLink
                  to="/admin?tab=user"
                  className={({ isActive }) => `
                    flex items-center gap-3 px-3 py-2 rounded-[var(--radius-sm)]
                    text-[13px] font-medium transition-all duration-150 group relative
                    ${(isActive && location.search.includes('tab=user'))
                      ? 'bg-[var(--primary-subtle)] text-[var(--primary)] border-l-2 border-[var(--primary)] pl-[10px]'
                      : 'text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text)]'
                    }
                  `}
                >
                  <User size={18} className={`flex-shrink-0 transition-opacity ${location.search.includes('tab=user') ? '' : 'opacity-60 group-hover:opacity-100'}`} />
                  {sidebarOpen && <span className="truncate">User</span>}
                  {!sidebarOpen && (
                    <div className="
                      absolute left-full ml-3 px-3 py-1.5 rounded-md
                      bg-[var(--bg-elevated)] border border-[var(--border)]
                      text-xs text-[var(--text)] whitespace-nowrap
                      opacity-0 pointer-events-none group-hover:opacity-100
                      transition-opacity duration-150 shadow-[var(--shadow-lg)] z-50
                    ">
                      User
                    </div>
                  )}
                </NavLink>
              </>
            ) : (
              <NavLink
                to={userRole === 'admin' ? '/admin?tab=user' : '/admin?tab=profile'}
                className={({ isActive }) => `
                  flex items-center gap-3 px-3 py-2 rounded-[var(--radius-sm)]
                  text-[13px] font-medium transition-all duration-150 group relative
                  ${isActive || location.pathname === '/admin'
                    ? 'bg-[var(--primary-subtle)] text-[var(--primary)] border-l-2 border-[var(--primary)] pl-[10px]'
                    : 'text-[var(--text-muted)] hover:bg-[var(--bg-hover)] hover:text-[var(--text)]'
                  }
                `}
              >
                <ShieldCheck size={18} className={`flex-shrink-0 transition-opacity ${location.pathname === '/admin' ? '' : 'opacity-60 group-hover:opacity-100'}`} />
                {sidebarOpen && <span className="truncate">{userRole === 'admin' ? 'Create User' : 'My Profile'}</span>}
                {!sidebarOpen && (
                  <div className="
                    absolute left-full ml-3 px-3 py-1.5 rounded-md
                    bg-[var(--bg-elevated)] border border-[var(--border)]
                    text-xs text-[var(--text)] whitespace-nowrap
                    opacity-0 pointer-events-none group-hover:opacity-100
                    transition-opacity duration-150 shadow-[var(--shadow-lg)] z-50
                  ">
                    {userRole === 'admin' ? 'Create User' : 'My Profile'}
                  </div>
                )}
              </NavLink>
            )}
          </div>
        </div>'''

# Replace from `        {/* User Management Dropdown */}` to the end of the block (before `</nav>`)
pattern = re.compile(r'\{\/\* User Management Dropdown \*\/\}.*?<\/div>\s*<\/div>\s*<\/nav>', re.DOTALL)
code = pattern.sub(new_block + '\n\n      </nav>', code)

with open('e:/aitrade/frontend/src/components/Sidebar.tsx', 'w', encoding='utf-8') as f:
    f.write(code)

print("done")
